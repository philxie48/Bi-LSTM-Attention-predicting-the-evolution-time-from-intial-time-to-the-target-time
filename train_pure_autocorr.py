import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from dataset import SequenceDataset
from streamlined_model import StreamlinedSequenceGenerator
from autocorrelation_loss import AutocorrelationLoss
import time
import numpy as np
import random
import json
from datetime import datetime
import pandas as pd
import types

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dataloaders(data_dir, batch_size=32, train_size=4400, val_size=550, transform=False, num_workers=4):
    """Create train, validation, and test dataloaders."""
    # Get all available files
    all_files = [os.path.join(data_dir, f"{i}.csv") for i in range(1, 5501) if os.path.exists(os.path.join(data_dir, f"{i}.csv"))]
    num_files = len(all_files)
    
    # Calculate split sizes
    test_size = num_files - train_size - val_size
    
    print(f"Data split: Training={train_size}, Validation={val_size}, Testing={test_size}")
    
    # Create indices for each split
    indices = list(range(1, num_files + 1))
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = SequenceDataset(data_dir, file_indices=train_indices, transform=transform)
    val_dataset = SequenceDataset(data_dir, file_indices=val_indices, transform=transform)
    test_dataset = SequenceDataset(data_dir, file_indices=test_indices, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Loaded {len(train_dataset)} training samples, {len(val_dataset)} validation samples, {len(test_dataset)} test samples")
    
    return train_loader, val_loader, test_loader, train_dataset

def get_teacher_forcing_ratio(epoch, total_epochs, start_ratio=0.7, end_ratio=0.6):
    """Get a teacher forcing ratio that decreases linearly with epochs."""
    return max(end_ratio, start_ratio - (start_ratio - end_ratio) * epoch / total_epochs)

def add_input_noise(inputs, noise_level=0.05):
    """Add small Gaussian noise to inputs for better generalization."""
    if noise_level <= 0:
        return inputs
    return inputs + noise_level * torch.randn_like(inputs)

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, epoch=0, total_epochs=1, clip_grad_norm=1.0, noise_level=0.05):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Get current teacher forcing ratio based on epoch
    if hasattr(model, 'use_teacher_forcing') and model.use_teacher_forcing:
        # For continued training, use the teacher_forcing_ratio directly
        if hasattr(model, '_continued_training') and model._continued_training:
            if hasattr(model, 'teacher_forcing_start') and hasattr(model, 'teacher_forcing_end'):
                current_tf_ratio = model.teacher_forcing_start - (model.teacher_forcing_start - model.teacher_forcing_end) * epoch / total_epochs
                model.teacher_forcing_ratio = current_tf_ratio
                print(f"Current teacher forcing ratio: {current_tf_ratio:.4f}")
            else:
                current_tf_ratio = model.teacher_forcing_ratio
                print(f"Current teacher forcing ratio: {current_tf_ratio:.4f}")
        else:
            current_tf_ratio = get_teacher_forcing_ratio(epoch, total_epochs)
            model.teacher_forcing_ratio = current_tf_ratio
            print(f"Current teacher forcing ratio: {current_tf_ratio:.4f}")
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Add small noise to inputs for better generalization
        inputs = add_input_noise(inputs, noise_level)
        
        # Convert to half precision if using AMP
        inputs = inputs.to(device, dtype=torch.float16)
        targets = targets.to(device, dtype=torch.float16)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # Backward pass with gradient scaling
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Track loss
        total_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    # Calculate average loss
    avg_loss = total_loss / len(train_loader)
    elapsed_time = time.time() - start_time
    
    print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.6f}, Time: {elapsed_time:.2f}s")
    
    return avg_loss

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    print(f"Validation - Avg Loss: {avg_loss:.6f}")
    
    return avg_loss

def test(model, test_loader, criterion, device):
    """Test the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    print(f"Test - Avg Loss: {avg_loss:.6f}")
    
    return avg_loss

def train_model(args):
    """Train the model with pure autocorrelation loss."""
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    args_dict = vars(args)
    args_file = os.path.join(args.output_dir, "args.json")
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, _ = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        train_size=4400,
        val_size=550,
        transform=False,
        num_workers=args.num_workers
    )
    
    # Load existing model if specified
    if args.existing_model is not None:
        print(f"Loading existing model from {args.existing_model}")
        checkpoint = torch.load(args.existing_model, map_location=device)
        model = StreamlinedSequenceGenerator(
            input_size=7,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_seq_len=151,
            dropout=args.dropout
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Enable teacher forcing with the correct ratio for the current epoch (101)
        if args.teacher_forcing:
            model.use_teacher_forcing = True
            # For continued training, we'll use teacher_forcing_start and teacher_forcing_end
            if hasattr(args, 'teacher_forcing_start') and hasattr(args, 'teacher_forcing_end'):
                model.teacher_forcing_start = args.teacher_forcing_start
                model.teacher_forcing_end = args.teacher_forcing_end
                model.teacher_forcing_ratio = args.teacher_forcing_start  # Initial value
                print(f"Setting teacher forcing ratio to decay from {args.teacher_forcing_start} to {args.teacher_forcing_end}")
            else:
                model.teacher_forcing_ratio = args.teacher_forcing_ratio
                print(f"Setting fixed teacher forcing ratio to {args.teacher_forcing_ratio}")
            model._continued_training = True  # Flag to indicate this is continued training
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.start_lr,  # Use the full learning rate for continued training
            weight_decay=args.weight_decay
        )
        
        # Don't load optimizer state, start fresh for the new learning rate schedule
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # For continued training, we'll start at epoch 0 but with the loaded model weights
        # This ensures we follow the new learning rate and teacher forcing schedules
        start_epoch = 0
        
        print(f"Loaded model weights from best checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        print(f"Starting new training phase at epoch 0 with updated parameters")
        
        # Create a new scheduler for the continued training
        warmup_epochs = int(args.epochs * 0.3)
        warmup_steps = warmup_epochs * len(train_loader)
        decay_steps = (args.epochs - warmup_epochs) * len(train_loader)
        
        # Define milestones for learning rate changes
        milestones = []
        lr_values = []
        
        # For continued training, we'll start directly at the specified learning rate
        # and then decay to the minimum learning rate
        for step in range(len(train_loader) * args.epochs + 1):
            milestone = step
            progress = step / (len(train_loader) * args.epochs)
            lr = args.start_lr - (args.start_lr - args.min_lr) * progress
            milestones.append(milestone)
            lr_values.append(lr)
        
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=1.0,  # Not used, we'll manually set the learning rate
        )
        
        # Override the scheduler's get_lr method to return our custom values
        def get_lr(self):
            current_step = self.last_epoch
            if current_step < len(lr_values):
                return [lr_values[current_step] for _ in self.base_lrs]
            return [args.min_lr for _ in self.base_lrs]
        
        # Apply the custom get_lr method
        scheduler.get_lr = types.MethodType(get_lr, scheduler)
        
        print("Learning rate scheduler configuration for continued training:")
        print(f"  - Starting learning rate: {args.start_lr}")
        print(f"  - Minimum learning rate: {args.min_lr}")
        print(f"  - Total epochs: {args.epochs}")
        print(f"  - Scheduler: Linear decay")
    else:
        # Create model
        model = StreamlinedSequenceGenerator(
            input_size=7,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_seq_len=151,
            dropout=args.dropout
        ).to(device)
        
        # Enable teacher forcing if specified
        if args.teacher_forcing:
            model.use_teacher_forcing = True
            model.teacher_forcing_ratio = args.teacher_forcing_ratio
        
        # Create optimizer with initial learning rate
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.start_lr / 10.0,  # Start with lower learning rate for warmup
            weight_decay=args.weight_decay
        )
        
        # Create a simple step-based scheduler
        # Phase 1: Linear warmup from start_lr/10 to start_lr for 30% of training
        # Phase 2: Linear decay from start_lr to min_lr for remaining 70% of training
        warmup_epochs = int(args.epochs * 0.3)
        warmup_steps = warmup_epochs * len(train_loader)
        decay_steps = (args.epochs - warmup_epochs) * len(train_loader)
        
        # Define milestones for learning rate changes
        milestones = []
        lr_values = []
        
        # Warmup phase
        for step in range(warmup_steps + 1):
            milestone = step
            lr = args.start_lr / 10.0 + (args.start_lr - args.start_lr / 10.0) * (step / warmup_steps)
            milestones.append(milestone)
            lr_values.append(lr)
        
        # Decay phase
        for step in range(1, decay_steps + 1):
            milestone = warmup_steps + step
            lr = args.start_lr - (args.start_lr - args.min_lr) * (step / decay_steps)
            milestones.append(milestone)
            lr_values.append(lr)
        
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=1.0,  # Not used, we'll manually set the learning rate
        )
        
        # Override the scheduler's get_lr method to return our custom values
        def get_lr(self):
            current_step = self.last_epoch
            if current_step < len(lr_values):
                return [lr_values[current_step] for _ in self.base_lrs]
            return [args.min_lr for _ in self.base_lrs]
        
        # Apply the custom get_lr method
        scheduler.get_lr = types.MethodType(get_lr, scheduler)
        
        print("Learning rate scheduler configuration:")
        print(f"  - Initial learning rate: {args.start_lr / 10.0}")
        print(f"  - Maximum learning rate: {args.start_lr}")
        print(f"  - Minimum learning rate: {args.min_lr}")
        print(f"  - Warmup epochs: {warmup_epochs}")
        print(f"  - Total epochs: {args.epochs}")
        print(f"  - Scheduler: Custom with warmup and linear decay")
        
        start_epoch = 0
    
    # Create criterion (pure autocorrelation loss)
    criterion = AutocorrelationLoss(max_lag=args.max_lag)
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler() if args.mixed_precision and torch.cuda.is_available() else None
    
    # Initialize training log
    log = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'lr': [],
        'epoch': []
    }
    
    # Train the model
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        # Train one epoch
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler=scheduler,
            epoch=epoch,
            total_epochs=args.epochs,
            clip_grad_norm=args.clip_grad_norm,
            noise_level=args.noise_level
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update log
        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        log['lr'].append(current_lr)
        log['epoch'].append(epoch + 1)
        
        # Print epoch summary
        print(f"Epoch: {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': args_dict
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'train_loss': train_loss,
                'args': args_dict
            }, best_model_path)
            print(f"Saved best model with validation loss {val_loss:.6f}")
    
    # Test the model
    test_loss = test(model, test_loader, criterion, device)
    log['test_loss'].append(test_loss)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'args': args_dict
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save training log
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)
    
    # Export training metrics to CSV
    export_metrics_to_csv(log, args.output_dir)
    
    # Save detailed epoch-by-epoch metrics CSV
    metrics_df = pd.DataFrame({
        'Epoch': log['epoch'],
        'Train_Loss': log['train_loss'],
        'Val_Loss': log['val_loss'],
        'Learning_Rate': log['lr']
    })
    
    # Save to CSV with epoch count in filename
    csv_file = os.path.join(args.output_dir, f"{len(log['epoch'])}.csv")
    metrics_df.to_csv(csv_file, index=False)
    print(f"Detailed metrics saved to {csv_file}")
    
    # Save component-specific losses for future use in time deduction
    component_losses = {}
    
    # Save best model with a consistent name for continuous training
    best_model_combined_path = os.path.join(args.output_dir, "best_model_combined.pth")
    if os.path.exists(best_model_path):
        # Copy the best model to the standard name for continuous training
        import shutil
        shutil.copy2(best_model_path, best_model_combined_path)
        print(f"Copied best model to {best_model_combined_path} for continuous training")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time / 60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    
    return model, best_val_loss, test_loss

def export_metrics_to_csv(log, output_dir):
    metrics_df = pd.DataFrame({
        'Epoch': log['epoch'],
        'Train Loss': log['train_loss'],
        'Validation Loss': log['val_loss'],
        'Test Loss': log['test_loss'],
        'Learning Rate': log['lr']
    })
    metrics_df.to_csv(os.path.join(output_dir, 'training_metrics.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence generator model with pure autocorrelation loss")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="D:/sample3", help="Directory containing data files")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
    # Loss parameters
    parser.add_argument("--max_lag", type=int, default=10, help="Maximum lag for autocorrelation calculation")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--start_lr", type=float, default=0.001, help="Starting learning rate")
    parser.add_argument("--min_lr", type=float, default=0.0005, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--teacher_forcing", action="store_true", help="Use teacher forcing")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.7, help="Teacher forcing ratio")
    parser.add_argument("--teacher_forcing_start", type=float, default=0.7, help="Teacher forcing start ratio")
    parser.add_argument("--teacher_forcing_end", type=float, default=0.6, help="Teacher forcing end ratio")
    parser.add_argument("--noise_level", type=float, default=0.05, help="Noise level for input perturbation")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save_every", type=int, default=10, help="Save model every N epochs")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="D:/neural network/100auto", help="Directory to save model checkpoints")
    
    # Model loading parameters
    parser.add_argument("--existing_model", type=str, default=None, help="Path to existing model to continue training from")
    
    args = parser.parse_args()
    
    # Set default parameters
    args.teacher_forcing = True  # Enable teacher forcing by default
    args.mixed_precision = True  # Enable mixed precision by default
    
    train_model(args)
