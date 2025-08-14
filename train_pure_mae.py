import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from dataset import SequenceDataset
from streamlined_model import StreamlinedSequenceGenerator
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
            current_tf_ratio = get_teacher_forcing_ratio(epoch, total_epochs, 
                                                        start_ratio=model.teacher_forcing_start, 
                                                        end_ratio=model.teacher_forcing_end)
            model.teacher_forcing_ratio = current_tf_ratio
            print(f"Current teacher forcing ratio: {current_tf_ratio:.4f}")
    
    # Create gradient scaler for mixed precision training
    scaler = GradScaler()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Add noise to inputs for better generalization
        inputs = add_input_noise(inputs, noise_level)
        
        optimizer.zero_grad()
        
        # Use mixed precision training if enabled
        if hasattr(model, 'mixed_precision') and model.mixed_precision:
            with autocast():
                outputs = model(inputs, targets)
                loss = criterion(outputs, targets)
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            
            # Clip gradients to prevent exploding gradients
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs, targets)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
        
        # Step the scheduler if provided (for LambdaLR scheduler)
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            # Print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{total_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{total_epochs} completed in {time.time() - start_time:.2f}s, Avg Loss: {avg_loss:.6f}")
    
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
    print(f"Validation Loss: {avg_loss:.6f}")
    
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
    print(f"Test Loss: {avg_loss:.6f}")
    
    return avg_loss

def train_model(args):
    """Train the model with pure MAE loss."""
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments to output directory
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_dataset = create_dataloaders(
        args.data_dir, args.batch_size, num_workers=args.num_workers
    )
    
    # Get input size from dataset
    input_size = train_dataset.sequences[0].shape[1] if train_dataset.sequences else 7
    seq_len = train_dataset.sequences[0].shape[0] if train_dataset.sequences else 151
    
    print(f"Input size: {input_size}, Sequence length: {seq_len}")
    
    # Initialize model
    if args.existing_model is not None and os.path.exists(args.existing_model):
        print(f"Loading existing model from {args.existing_model}")
        checkpoint = torch.load(args.existing_model, map_location=device)
        
        # Create a new model with the same architecture
        model = StreamlinedSequenceGenerator(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_seq_len=seq_len,
            dropout=args.dropout
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set teacher forcing parameters for continued training
        if args.teacher_forcing:
            model.use_teacher_forcing = True
            model.teacher_forcing_start = args.teacher_forcing_start
            model.teacher_forcing_end = args.teacher_forcing_end
            model.teacher_forcing_ratio = args.teacher_forcing_start
            model._continued_training = True
            print(f"Set teacher forcing for continued training: {args.teacher_forcing_start} to {args.teacher_forcing_end}")
        
        # Set mixed precision flag
        model.mixed_precision = args.mixed_precision
        
        # Get previous best validation loss
        previous_best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Previous best validation loss: {previous_best_val_loss:.6f}")
        
        print(f"Model loaded successfully. Starting from epoch {checkpoint.get('epoch', 0) + 1}")
    else:
        print("Initializing new model")
        model = StreamlinedSequenceGenerator(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_seq_len=seq_len,
            dropout=args.dropout
        )
        
        # Set teacher forcing parameters
        if args.teacher_forcing:
            model.use_teacher_forcing = True
            model.teacher_forcing_start = args.teacher_forcing_start
            model.teacher_forcing_end = args.teacher_forcing_end
            model.teacher_forcing_ratio = args.teacher_forcing_start
            print(f"Set teacher forcing: {args.teacher_forcing_start} to {args.teacher_forcing_end}")
        
        # Set mixed precision flag
        model.mixed_precision = args.mixed_precision
    
    model = model.to(device)
    
    # Define loss function (MAE Loss)
    criterion = nn.L1Loss()
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.start_lr, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    if args.existing_model is not None and os.path.exists(args.existing_model):
        # For continued training, create a custom scheduler
        class CustomScheduler:
            def __init__(self, optimizer, start_lr, min_lr, epochs):
                self.optimizer = optimizer
                self.start_lr = start_lr
                self.min_lr = min_lr
                self.epochs = epochs
                self.current_epoch = 0
                
                # Set initial learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.start_lr
            
            def step(self):
                self.current_epoch += 1
                lr = self.start_lr - (self.start_lr - self.min_lr) * self.current_epoch / self.epochs
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            def get_lr(self):
                return [param_group['lr'] for param_group in self.optimizer.param_groups]
        
        scheduler = CustomScheduler(optimizer, args.start_lr, args.min_lr, args.epochs)
    else:
        # For new training, implement a simple warmup + decay scheduler directly
        # Set initial learning rate to warmup start (50% of max)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.start_lr * 0.5
        
        # Create a scheduler class that handles both warmup and decay
        class WarmupCosineScheduler:
            def __init__(self, optimizer, warmup_start_lr, max_lr, min_lr, total_steps, warmup_steps):
                self.optimizer = optimizer
                self.warmup_start_lr = warmup_start_lr
                self.max_lr = max_lr
                self.min_lr = min_lr
                self.total_steps = total_steps
                self.warmup_steps = warmup_steps
                self.current_step = 0
            
            def step(self):
                self.current_step += 1
                if self.current_step < self.warmup_steps:
                    # Linear warmup
                    progress = self.current_step / self.warmup_steps
                    lr = self.warmup_start_lr + progress * (self.max_lr - self.warmup_start_lr)
                else:
                    # Cosine decay
                    progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                    lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
                
                # Update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            def get_lr(self):
                return [param_group['lr'] for param_group in self.optimizer.param_groups]
        
        # Calculate total steps and warmup steps
        total_steps = len(train_loader) * args.epochs
        warmup_steps = int(total_steps * 0.3)  # 30% warmup
        
        # Create scheduler
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_start_lr=args.start_lr * 0.5,  # Start at 50% of max LR
            max_lr=args.start_lr,
            min_lr=args.min_lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps
        )
    
    # Training loop
    best_val_loss = float('inf')
    if args.existing_model is not None and os.path.exists(args.existing_model):
        # Use previous best validation loss when continuing training
        checkpoint = torch.load(args.existing_model, map_location=device)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Using previous best validation loss as baseline: {best_val_loss:.6f}")
        
        # Copy the existing best model to the new output directory to preserve it
        import shutil
        best_model_path = os.path.join(args.output_dir, "best_model.pth")
        if not os.path.exists(best_model_path):
            print(f"Copying existing best model to {best_model_path}")
            shutil.copy2(args.existing_model, best_model_path)
    
    start_time = time.time()
    
    # Initialize log dictionary
    log = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'lr': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scheduler=scheduler if args.existing_model is None else None,  # Only use scheduler for new training
            epoch=epoch, total_epochs=args.epochs,
            clip_grad_norm=args.clip_grad_norm,
            noise_level=args.noise_level
        )
        
        # Step the custom scheduler for continued training
        if args.existing_model is not None:
            scheduler.step()
            # Print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.6f}")
        
        # Validate the model
        val_loss = validate(model, val_loader, criterion, device)
        
        # Test the model
        test_loss = test(model, test_loader, criterion, device)
        
        # Save current learning rate
        current_lr = scheduler.get_lr()[0]
        
        # Update log
        log['epoch'].append(epoch + 1)
        log['train_loss'].append(train_loss)
        log['val_loss'].append(val_loss)
        log['test_loss'].append(test_loss)
        log['lr'].append(current_lr)
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'test_loss': test_loss,
                'args': vars(args)
            }, best_model_path)
            
            print(f"Saved best model to {best_model_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'test_loss': test_loss,
                'args': vars(args)
            }, checkpoint_path)
            
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Export metrics to CSV after each epoch
        export_metrics_to_csv(log, args.output_dir)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'test_loss': test_loss,
        'args': vars(args)
    }, final_model_path)
    
    print(f"Saved final model to {final_model_path}")
    
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
    parser = argparse.ArgumentParser(description="Train a sequence generator model with pure MAE loss")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="D:/sample3", help="Directory containing data files")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    # Model parameters
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    
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
    parser.add_argument("--output_dir", type=str, default="D:/neural network/100mae", help="Directory to save model checkpoints")
    
    # Model loading parameters
    parser.add_argument("--existing_model", type=str, default=None, help="Path to existing model to continue training from")
    
    args = parser.parse_args()
    
    # Set default parameters
    args.teacher_forcing = True  # Enable teacher forcing by default
    args.mixed_precision = True  # Enable mixed precision by default
    
    train_model(args)
