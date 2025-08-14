# Neural Network Training Documentation

This document describes the complete training pipeline for sequence generation models using different loss function strategies. The training system supports three different approaches: pure MAE loss, pure autocorrelation loss, and combined loss functions.

## Overview

The training pipeline consists of three main training scripts and corresponding batch files for automated execution:

1. **`train_pure_mae.py`** - Pure Mean Absolute Error (MAE) loss training
2. **`train_pure_autocorr.py`** - Pure autocorrelation loss training  
3. **`train_combined_mae_autocorr.py`** - Combined MAE + autocorrelation loss training

Each training approach has corresponding batch files for easy execution and parameter management.

---

## Training Script Overview

### Common Features

All training scripts share these core features:

- **Model Architecture**: StreamlinedSequenceGenerator with LSTM layers
- **Mixed Precision Training**: For faster GPU computation and memory efficiency
- **Teacher Forcing**: Gradual decay from 70% to 60% over training epochs
- **Learning Rate Scheduling**: Warmup + cosine/linear decay
- **Gradient Clipping**: Prevents exploding gradients (norm=1.0)
- **Input Noise**: Gaussian noise injection for better generalization
- **Checkpointing**: Saves best model, regular checkpoints, and final model
- **Metrics Export**: CSV files with training metrics for analysis

### Data Pipeline

- **Input Data**: CSV files from `D:/sample3/` (output of sample production pipeline)
- **Data Split**: 4400 train / 550 validation / remaining test samples
- **Sequence Format**: 151 timesteps × 7 features (step + 5 PCA components + 2 parameters)
- **Batch Processing**: Configurable batch size (default: 32)

---

## 1. Pure MAE Training (`train_pure_mae.py`)

### Purpose
Trains the model using only Mean Absolute Error (L1) loss for point-wise accuracy.

### Loss Function
```python
criterion = nn.L1Loss()  # Simple L1/MAE loss
```

### Key Features
- **Focus**: Accurate point-wise predictions
- **Strengths**: Simple, stable training, good for absolute value accuracy
- **Weaknesses**: May not preserve temporal patterns well

### Training Configuration
- **Default Output**: `D:/neural network/100mae`
- **Loss Tracking**: Single MAE loss value
- **Save Frequency**: Every 10 epochs

### Usage
```bash
python train_pure_mae.py --epochs 100 --start_lr 0.001 --output_dir "D:/neural network/100mae"
```

### Parameters
- `--hidden_size 256`: LSTM hidden dimension
- `--num_layers 2`: Number of LSTM layers
- `--dropout 0.3`: Dropout rate
- `--batch_size 32`: Training batch size
- `--start_lr 0.001`: Initial learning rate
- `--min_lr 0.0005`: Minimum learning rate
- `--teacher_forcing`: Enable teacher forcing (default: True)
- `--mixed_precision`: Enable mixed precision training (default: True)

---

## 2. Pure Autocorrelation Training (`train_pure_autocorr.py`)

### Purpose
Trains the model using only autocorrelation loss to preserve temporal patterns and correlations.

### Loss Function
```python
criterion = AutocorrelationLoss(max_lag=args.max_lag)  # Temporal pattern preservation
```

### Key Features
- **Focus**: Temporal pattern preservation and autocorrelation structure
- **Strengths**: Maintains realistic temporal dynamics and patterns
- **Weaknesses**: May sacrifice point-wise accuracy for pattern preservation

### Training Configuration
- **Default Output**: `D:/neural network/100auto`
- **Max Lag**: 10 timesteps for autocorrelation calculation
- **Loss Tracking**: Single autocorrelation loss value
- **Save Frequency**: Every 10 epochs

### Usage
```bash
python train_pure_autocorr.py --epochs 100 --max_lag 10 --output_dir "D:/neural network/100auto"
```

### Specific Parameters
- `--max_lag 10`: Maximum lag for autocorrelation calculation
- **Advanced Learning Rate**: Custom warmup + linear decay scheduler
- **Mixed Precision**: Enhanced with gradient scaling

---

## 3. Combined Training (`train_combined_mae_autocorr.py`)

### Purpose
Trains the model using a weighted combination of MAE and autocorrelation losses to balance point-wise accuracy with temporal pattern preservation.

### Loss Function
```python
class CombinedMAEAutocorrLoss(nn.Module):
    def __init__(self, alpha=0.5, max_lag=10):
        self.alpha = alpha  # 50% MAE + 50% Autocorrelation
        self.mae_loss = nn.L1Loss()
        self.autocorr_loss = AutocorrelationLoss(max_lag=max_lag)
    
    def forward(self, y_pred, y_true):
        mae = self.mae_loss(y_pred, y_true)
        autocorr = self.autocorr_loss(y_pred, y_true)
        combined = self.alpha * mae + (1 - self.alpha) * autocorr
        return combined, mae, autocorr
```

### Key Features
- **Focus**: Balanced approach combining accuracy and pattern preservation
- **Loss Weighting**: 50% MAE + 50% Autocorrelation (configurable)
- **Detailed Tracking**: Separate tracking of MAE, autocorrelation, and combined losses
- **Best of Both**: Combines strengths of both approaches

### Training Configuration
- **Default Output**: `D:/neural network/50mae_50auto`
- **Loss Components**: Combined, MAE, and autocorrelation losses tracked separately
- **Save Frequency**: Every 5 epochs (more frequent due to complexity)

### Usage
```bash
python train_combined_mae_autocorr.py --epochs 100 --max_lag 10 --output_dir "D:/neural network/50mae_50auto"
```

---

## Batch File Automation

### 500-Epoch Training Strategy

The training system implements a 500-epoch strategy divided into 5 phases of 100 epochs each, with progressive parameter tuning:

#### Phase 1: `run_combined_mae_auto_1_100.bat`
```batch
python train_combined_mae_autocorr.py ^
    --data_dir "D:/sample3" ^
    --output_dir "D:/neural network/50mae_50auto_1_100" ^
    --epochs 100 ^
    --start_lr 0.001 ^
    --min_lr 0.0005 ^
    --teacher_forcing_start 0.7 ^
    --teacher_forcing_end 0.6 ^
    --max_lag 10 ^
    --noise_level 0.05
```

#### Progressive Training Parameters

The 500-epoch training strategy uses carefully tuned parameter progression across 5 phases:

**Pure MAE Training Strategy:**
| Phase | Epochs | Learning Rate | Teacher Forcing | Noise Level | Weight Decay | Grad Clip |
|-------|--------|---------------|-----------------|-------------|--------------|-----------|
| 1     | 1-100  | 0.001→0.0005  | 0.7→0.6         | 0.05        | 0.01         | 1.0       |
| 2     | 101-200| 0.0005→0.0001 | 0.6→0.5         | 0.04        | 0.008        | 0.9       |
| 3     | 201-300| 0.0001→0.00005| 0.5→0.4         | 0.03        | 0.005        | 0.8       |
| 4     | 301-400| 0.00005→0.00001| 0.4→0.3        | 0.02        | 0.003        | 0.7       |
| 5     | 401-500| 0.00001→0.000005| 0.3→0.2       | 0.01        | 0.002        | 0.6       |

**Pure Autocorrelation Training Strategy:**
| Phase | Epochs | Learning Rate | Teacher Forcing | Max Lag | Noise Level | Weight Decay | Grad Clip |
|-------|--------|---------------|-----------------|---------|-------------|--------------|-----------|
| 1     | 1-100  | 0.001→0.0005  | 0.7→0.6         | 10      | 0.05        | 0.01         | 1.0       |
| 2     | 101-200| 0.0005→0.0001 | 0.6→0.5         | 15      | 0.05        | 0.008        | 1.0       |
| 3     | 201-300| 0.0001→0.00005| 0.5→0.4         | 15      | 0.04        | 0.005        | 0.9       |
| 4     | 301-400| 0.00005→0.00001| 0.4→0.3        | 20      | 0.02        | 0.003        | 0.7       |
| 5     | 401-500| 0.00001→0.000005| 0.3→0.2       | 20      | 0.01        | 0.002        | 0.6       |

**Combined MAE+Autocorrelation Training Strategy:**
| Phase | Epochs | Learning Rate | Teacher Forcing | Max Lag | Noise Level | Weight Decay | Grad Clip |
|-------|--------|---------------|-----------------|---------|-------------|--------------|-----------|
| 1     | 1-100  | 0.001→0.0005  | 0.7→0.6         | 10      | 0.05        | 0.01         | 1.0       |
| 2     | 101-200| 0.0005→0.0001 | 0.6→0.5         | 15      | 0.04        | 0.008        | 0.9       |
| 3     | 201-300| 0.0001→0.00005| 0.5→0.4         | 15      | 0.03        | 0.005        | 0.8       |
| 4     | 301-400| 0.00005→0.00001| 0.4→0.3        | 20      | 0.02        | 0.003        | 0.7       |
| 5     | 401-500| 0.00001→0.000005| 0.3→0.2       | 20      | 0.01        | 0.002        | 0.6       |

#### Parameter Progression Strategy Rationale

**Learning Rate Decay:**
- **Phase 1**: Start with moderate learning rate (0.001) for initial convergence
- **Phases 2-5**: Exponential decay (÷10 each phase) for fine-tuning and stability
- **Final phase**: Ultra-low learning rate (0.00001→0.000005) for precise optimization

**Teacher Forcing Schedule:**
- **Early phases**: High teacher forcing (0.7) provides strong guidance
- **Progressive reduction**: Gradually reduce dependency on ground truth
- **Final phase**: Minimal teacher forcing (0.3→0.2) forces autonomous generation

**Noise Level Reduction:**
- **Phase 1**: High noise (0.05) for robust feature learning
- **Progressive reduction**: Decrease noise as model stabilizes
- **Final phase**: Minimal noise (0.01) for precise predictions

**Weight Decay Strategy:**
- **Early phases**: Strong regularization (0.01) prevents early overfitting
- **Progressive reduction**: Relaxed regularization as learning stabilizes
- **Final phase**: Minimal regularization (0.002) allows fine-tuning

**Gradient Clipping Adaptation:**
- **Early phases**: Strong clipping (1.0) prevents gradient explosion
- **Progressive reduction**: Gradual relaxation as training stabilizes
- **Final phase**: Minimal clipping (0.6) preserves gradient information

**Max Lag Evolution (Autocorrelation-based methods):**
- **Phase 1**: Short-range correlations (lag=10) for basic pattern learning
- **Phase 2-3**: Medium-range correlations (lag=15) for temporal structure
- **Phase 4-5**: Long-range correlations (lag=20) for complex dependencies

#### Complete Batch File Sets

**Pure MAE Training (100% MAE Loss):**
- `run_pure_mae_1_100.bat` - Phase 1: Initial training
- `run_pure_mae_101_200.bat` - Phase 2: Continued training  
- `run_pure_mae_201_300.bat` - Phase 3: Continued training
- `run_pure_mae_301_400.bat` - Phase 4: Continued training
- `run_pure_mae_401_500.bat` - Phase 5: Final training

**Pure Autocorrelation Training (100% Autocorr Loss):**
- `run_pure_autocorr_1_100.bat` - Phase 1: Initial training
- `run_pure_autocorr_101_200.bat` - Phase 2: Continued training
- `run_pure_autocorr_201_300.bat` - Phase 3: Continued training
- `run_pure_autocorr_301_400.bat` - Phase 4: Continued training
- `run_pure_autocorr_401_500.bat` - Phase 5: Final training

**Combined Training (50% MAE + 50% Autocorr):**
- `run_combined_mae_auto_1_100.bat` - Phase 1: Initial training
- `run_combined_mae_auto_101_200.bat` - Phase 2: Continued training
- `run_combined_mae_auto_201_300.bat` - Phase 3: Continued training
- `run_combined_mae_auto_301_400.bat` - Phase 4: Continued training
- `run_combined_mae_auto_401_500.bat` - Phase 5: Final training

#### Example Phase Progression

**Phase 1 → Phase 2 Transition (Pure MAE):**
```batch
# Phase 1
run_pure_mae_1_100.bat
# Creates: D:/neural network/100mae_1_100/best_model.pth

# Phase 2  
run_pure_mae_101_200.bat
# Uses: --existing_model "D:/neural network/100mae_1_100/best_model.pth"
# Creates: D:/neural network/100mae_101_200/best_model.pth
```

**Phase 4 → Phase 5 Transition (Combined):**
```batch
# Phase 4
run_combined_mae_auto_301_400.bat
# Parameters: --start_lr 0.00005 --teacher_forcing_start 0.4 --max_lag 20

# Phase 5
run_combined_mae_auto_401_500.bat  
# Parameters: --start_lr 0.00001 --teacher_forcing_start 0.3 --max_lag 20
# Uses: --existing_model "D:/neural network/50mae_50auto_301_400/best_model.pth"
```

### Batch File Features

#### Automated Process Management
```batch
@echo off
REM Kill any existing Python processes
taskkill /f /im python.exe 2>nul

REM Run training with specific parameters
python train_combined_mae_autocorr.py ^
    [parameters...]

REM Display next phase instructions
echo To continue training for the next 100 epochs (Phase 2), use:
echo python train_combined_mae_autocorr.py --existing_model "[path]" [next_phase_params]
pause
```

#### Continuation Instructions
Each batch file provides exact commands for continuing to the next phase, ensuring seamless progressive training.

---

## Training Workflow

### 1. Complete 500-Epoch Training Workflows

**Pure MAE Training (Complete 500-epoch sequence):**
```batch
# Phase 1: Initial training (epochs 1-100)
run_pure_mae_1_100.bat

# Phase 2: Continued training (epochs 101-200)
run_pure_mae_101_200.bat

# Phase 3: Continued training (epochs 201-300)
run_pure_mae_201_300.bat

# Phase 4: Continued training (epochs 301-400)
run_pure_mae_301_400.bat

# Phase 5: Final training (epochs 401-500)
run_pure_mae_401_500.bat
```

**Pure Autocorrelation Training (Complete 500-epoch sequence):**
```batch
# Phase 1: Initial training (epochs 1-100)
run_pure_autocorr_1_100.bat

# Phase 2: Continued training (epochs 101-200)
run_pure_autocorr_101_200.bat

# Phase 3: Continued training (epochs 201-300)
run_pure_autocorr_201_300.bat

# Phase 4: Continued training (epochs 301-400)
run_pure_autocorr_301_400.bat

# Phase 5: Final training (epochs 401-500)
run_pure_autocorr_401_500.bat
```

**Combined Training (Complete 500-epoch sequence):**
```batch
# Phase 1: Initial training (epochs 1-100)
run_combined_mae_auto_1_100.bat

# Phase 2: Continued training (epochs 101-200)
run_combined_mae_auto_101_200.bat

# Phase 3: Continued training (epochs 201-300)
run_combined_mae_auto_201_300.bat

# Phase 4: Continued training (epochs 301-400)
run_combined_mae_auto_301_400.bat

# Phase 5: Final training (epochs 401-500)
run_combined_mae_auto_401_500.bat
```

### 2. Single Phase Training (Alternative approach)

**For Pure MAE:**
```bash
run_pure_mae_1_100.bat
```

**For Pure Autocorrelation:**
```bash
run_pure_autocorr_1_100.bat  
```

**For Combined Approach:**
```bash
run_combined_mae_auto_1_100.bat
```

### 3. Manual Continued Training (Custom Parameters)

The system supports loading existing models and continuing training with custom parameters:

```bash
python train_combined_mae_autocorr.py \
    --existing_model "D:/neural network/50mae_50auto_1_100/best_model.pth" \
    --epochs 100 \
    --start_lr 0.0005 \
    --min_lr 0.0001 \
    --output_dir "D:/neural network/50mae_50auto_101_200" \
    --teacher_forcing_start 0.6 \
    --teacher_forcing_end 0.5
```

### 4. Model Progression and Output Directories

Each training phase saves multiple model files in dedicated directories:

**Pure MAE Training Output Structure:**
```
D:/neural network/
├── 100mae_1_100/          # Phase 1 (epochs 1-100)
├── 100mae_101_200/        # Phase 2 (epochs 101-200)
├── 100mae_201_300/        # Phase 3 (epochs 201-300)
├── 100mae_301_400/        # Phase 4 (epochs 301-400)
└── 100mae_401_500/        # Phase 5 (epochs 401-500)
```

**Pure Autocorrelation Training Output Structure:**
```
D:/neural network/
├── 100auto_1_100/         # Phase 1 (epochs 1-100)
├── 100auto_101_200/       # Phase 2 (epochs 101-200)
├── 100auto_201_300/       # Phase 3 (epochs 201-300)
├── 100auto_301_400/       # Phase 4 (epochs 301-400)
└── 100auto_401_500/       # Phase 5 (epochs 401-500)
```

**Combined Training Output Structure:**
```
D:/neural network/
├── 50mae_50auto_1_100/    # Phase 1 (epochs 1-100)
├── 50mae_50auto_101_200/  # Phase 2 (epochs 101-200)
├── 50mae_50auto_201_300/  # Phase 3 (epochs 201-300)
├── 50mae_50auto_301_400/  # Phase 4 (epochs 301-400)
└── 50mae_50auto_401_500/  # Phase 5 (epochs 401-500)
```

**Each Phase Directory Contains:**
- **`best_model.pth`**: Best validation loss model
- **`final_model.pth`**: Final epoch model  
- **`best_model_combined.pth`**: Standardized name for continuous training
- **`checkpoint_epoch_N.pth`**: Regular checkpoints (every 5 epochs)
- **`training_metrics.csv`**: Detailed loss and learning rate metrics
- **`args.json`**: Training configuration parameters

---

## Output Structure

### Training Results Directory

```
D:/neural network/[approach_name]/
├── args.json                    # Training arguments
├── best_model.pth              # Best validation model
├── final_model.pth             # Final model
├── best_model_combined.pth     # For continuous training
├── checkpoint_epoch_5.pth      # Regular checkpoints
├── checkpoint_epoch_10.pth
├── ...
└── training_metrics.csv        # Detailed metrics
```

### Metrics Tracking

**Pure MAE Training:**
```csv
Epoch,Train Loss,Validation Loss,Test Loss,Learning Rate
1,0.045623,0.043821,0.044125,0.0005
2,0.042156,0.041234,0.041876,0.0007
...
```

**Combined Training:**
```csv
Epoch,Train Loss,Train MAE Loss,Train Autocorr Loss,Validation Loss,Validation MAE Loss,Validation Autocorr Loss,Test Loss,Test MAE Loss,Test Autocorr Loss,Learning Rate
1,0.045623,0.023456,0.022167,0.043821,0.022134,0.021687,0.044125,0.022543,0.021582,0.0005
...
```

---

## Advanced Features

### Teacher Forcing Schedule

Teacher forcing ratio decreases linearly over epochs:
```python
def get_teacher_forcing_ratio(epoch, total_epochs, start_ratio=0.7, end_ratio=0.6):
    return max(end_ratio, start_ratio - (start_ratio - end_ratio) * epoch / total_epochs)
```

### Learning Rate Scheduling

#### New Training: Warmup + Cosine Decay
```python
class WarmupCosineScheduler:
    - Phase 1: Linear warmup (30% of training)
    - Phase 2: Cosine decay (70% of training)
```

#### Continued Training: Linear Decay
```python
class CustomScheduler:
    - Linear decay from start_lr to min_lr over epochs
```

### Mixed Precision Training

```python
# Automatic mixed precision for faster training
with autocast():
    outputs = model(inputs, targets)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Input Noise Injection

```python
def add_input_noise(inputs, noise_level=0.05):
    return inputs + noise_level * torch.randn_like(inputs)
```

---

## Performance Optimization

### GPU Utilization
- **Mixed Precision**: Reduces memory usage and increases speed
- **Gradient Scaling**: Prevents underflow in FP16 training
- **Optimized Data Loading**: Multi-worker data loading (default: 4 workers)

### Memory Management
- **Gradient Accumulation**: Effective batch size scaling
- **Checkpoint Offloading**: Saves memory during validation
- **Dynamic Memory**: Automatic memory management

### Training Speed
- **Batch Processing**: Optimized batch sizes for GPU utilization
- **Efficient Schedulers**: Minimal overhead learning rate updates
- **Fast Data Pipeline**: Pre-loaded sequences with caching

---

## Model Selection Strategy

### Choosing Training Approach

**Use Pure MAE when:**
- Point-wise accuracy is most important
- Simple, stable training is preferred
- Limited computational resources

**Use Pure Autocorrelation when:**
- Temporal patterns are critical
- Realistic dynamics are more important than exact values
- Working with highly correlated time series

**Use Combined Approach when:**
- Both accuracy and patterns are important
- Best overall performance is needed
- Sufficient computational resources available

### Hyperparameter Tuning

**Critical Parameters:**
- `max_lag`: Higher values capture longer-range correlations
- `alpha`: Balance between MAE and autocorrelation (0.5 = equal weight)
- `teacher_forcing_ratio`: Higher values provide more guidance
- `noise_level`: Higher values improve generalization but may hurt accuracy

**Training Schedule:**
- Start with higher learning rates and teacher forcing
- Gradually reduce both over training phases
- Increase max_lag for longer-range pattern capture
- Decrease noise level as training progresses

---

## Troubleshooting

### Common Issues

**Memory Issues:**
- Reduce batch size (`--batch_size 16`)
- Disable mixed precision (`--no_mixed_precision`)
- Reduce number of workers (`--num_workers 2`)

**Training Instability:**
- Increase gradient clipping (`--clip_grad_norm 2.0`)
- Reduce learning rate (`--start_lr 0.0005`)
- Adjust loss weighting (`--alpha 0.3` for more autocorr emphasis)

**Poor Convergence:**
- Check data quality and preprocessing
- Verify teacher forcing schedule
- Adjust learning rate schedule
- Increase model capacity (`--hidden_size 512`)

### Monitoring Training

**Key Metrics to Watch:**
- Validation loss convergence
- Training/validation loss gap (overfitting)
- Learning rate schedule effectiveness
- Component loss balance (for combined training)

**Early Stopping Criteria:**
- Validation loss plateau for >10 epochs
- Severe overfitting (train/val gap >50%)
- Gradient explosion (despite clipping)

---

## Results Analysis

### Model Evaluation

**Quantitative Metrics:**
- Final validation/test loss
- Component loss breakdown (MAE vs autocorrelation)
- Training convergence speed
- Memory and time efficiency

**Qualitative Assessment:**
- Generated sequence realism
- Temporal pattern preservation
- Parameter response accuracy
- Generalization to unseen data

### Comparison Framework

Use the exported CSV metrics to compare different approaches:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics from different training runs
mae_metrics = pd.read_csv('D:/neural network/100mae/training_metrics.csv')
auto_metrics = pd.read_csv('D:/neural network/100auto/training_metrics.csv') 
combined_metrics = pd.read_csv('D:/neural network/50mae_50auto/training_metrics.csv')

# Plot comparison
plt.figure(figsize=(12, 8))
plt.plot(mae_metrics['Epoch'], mae_metrics['Validation Loss'], label='Pure MAE')
plt.plot(auto_metrics['Epoch'], auto_metrics['Validation Loss'], label='Pure Autocorr')
plt.plot(combined_metrics['Epoch'], combined_metrics['Validation Loss'], label='Combined')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Training Approach Comparison')
plt.show()
```

This comprehensive training system provides flexibility to experiment with different loss functions while maintaining consistent training infrastructure and progressive learning strategies.


