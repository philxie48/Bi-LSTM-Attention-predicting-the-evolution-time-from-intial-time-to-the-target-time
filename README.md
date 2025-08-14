# Phase Field Neural Sequence Generation

A comprehensive pipeline for generating neural network training data from phase field simulations and training sequence generation models with different loss function strategies.

## 🎯 Project Overview

This project implements a complete machine learning pipeline that:

1. **Generates Physics-Based Data**: Uses Cahn-Hilliard equation to simulate spinodal decomposition
2. **Extracts Meaningful Features**: Applies PCA to spatial autocorrelation functions
3. **Trains Neural Networks**: Implements three different training strategies for sequence generation
4. **Predicts Temporal Evolution**: Generate complete sequences from initial conditions
5. **Performs Time Deduction**: Find most likely time positions for given target states
6. **Provides Comprehensive Analysis**: Includes detailed documentation and parameter optimization

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Training Strategies](#training-strategies)
- [Model Prediction](#model-prediction)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd phase-field-neural-sequence-generation
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Quick Demo

Run a complete mini-pipeline:

```bash
# 1. Generate sample data (10 simulations)
cd sample/
python phasefield.py  # Edit line 148 to set batch name

# 2. Process with PCA
python AutoandPCA.py  # Enter batch name when prompted

# 3. Extract and convert data
python extract_pca_files.py
python "trans_npz to csv.py"

# 4. Train a model (100 epochs)
cd ../sequence_generator/
run_combined_mae_auto_1_100.bat
```

## 💻 Installation

### System Requirements

**Minimum Requirements:**
- OS: Windows 10/Linux Ubuntu 18.04+/macOS 10.15+
- RAM: 8GB
- Storage: 20GB free space
- GPU: Any CUDA-capable GPU (optional but recommended)

**Recommended Requirements:**
- RAM: 16GB+
- Storage: 50GB+ free space
- GPU: NVIDIA RTX 3060 or better
- CPU: 8+ cores for data processing

### Python Dependencies

Create `requirements.txt`:

```txt
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0

# Machine learning
torch>=1.12.0
torchvision>=0.13.0
scikit-learn>=1.0.0

# Data processing
vtk>=9.1.0
h5py>=3.6.0

# Visualization (optional)
seaborn>=0.11.0
plotly>=5.0.0

# Development tools (optional)
jupyter>=1.0.0
tqdm>=4.62.0
```

### CUDA Setup (GPU Training)

**For NVIDIA GPUs:**

1. **Install CUDA Toolkit 11.6+:**
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

2. **Verify CUDA installation:**
```bash
nvcc --version
nvidia-smi
```

3. **Install PyTorch with CUDA:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. **Test GPU availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name()}")
```

## 📁 Project Structure

```
phase-field-neural-sequence-generation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── realization.md                     # Technical implementation details
├── sample/                            # Data generation pipeline
│   ├── sample_produce.md              # Data pipeline documentation
│   ├── phasefield.py                  # Phase field simulation
│   ├── AutoandPCA.py                  # PCA analysis
│   ├── extract_pca_files.py           # Data extraction
│   ├── trans_npz to csv.py            # NPZ to CSV conversion
│   └── simple_npz_to_csv.py           # Alternative converter
├── sequence_generator/                # Neural network training
│   ├── training.md                    # Training documentation
│   ├── train_pure_mae.py              # Pure MAE loss training
│   ├── train_pure_autocorr.py         # Pure autocorr loss training
│   ├── train_combined_mae_autocorr.py # Combined loss training
│   ├── streamlined_model.py           # LSTM model architecture
│   ├── dataset.py                     # Data loading utilities
│   ├── autocorrelation_loss.py        # Custom loss functions
│   ├── run_*_*.bat                    # Training batch files
│   ├── run_custom_prediction.py       # Sequence generation from initial conditions
│   ├── simple_time_deduction.py       # Time position deduction
│   ├── prediction.md                  # Prediction usage documentation
│   └── mae100/                        # Additional utilities
├── models/                            # Saved model checkpoints
├── data/                              # Generated datasets
└── results/                           # Training results and analysis
```

## 📖 Usage Guide

### Step 1: Generate Simulation Data

**Configure simulation parameters in `sample/phasefield.py`:**

```python
# Edit line 148 to set batch name
main_simulation("your_batch_name")

# Simulation parameters (lines 113-118):
c0 = random.uniform(0.3, 0.7)        # Initial concentration
mobility = random.uniform(1, 2)      # Mobility parameter  
grad_coef = random.uniform(0.5, 1)   # Gradient coefficient
noise_amp = random.uniform(0.05, 0.15)  # Noise amplitude
```

**Run simulation:**
```bash
cd sample/
python phasefield.py
```

**Output:** VTK files in `D:/sample/{batch_name}/`

### Step 2: Process with PCA Analysis

**Run PCA analysis:**
```bash
python AutoandPCA.py
# Enter batch name when prompted
```

**Output:** PCA results in `D:/sample1/{batch_name}/`

### Step 3: Prepare Training Data

**Extract and convert data:**
```bash
python extract_pca_files.py
python "trans_npz to csv.py"
```

**Output:** CSV files in `D:/sample3/` ready for training

### Step 4: Train Neural Networks

**Choose training strategy:**

**Option A: Combined Training (Recommended)**
```bash
cd sequence_generator/
run_combined_mae_auto_1_100.bat  # Phase 1
run_combined_mae_auto_101_200.bat  # Phase 2
# ... continue for all 5 phases
```

**Option B: Pure MAE Training**
```bash
run_pure_mae_1_100.bat
run_pure_mae_101_200.bat
# ... continue for all 5 phases
```

**Option C: Pure Autocorrelation Training**
```bash
run_pure_autocorr_1_100.bat
run_pure_autocorr_101_200.bat
# ... continue for all 5 phases
```

## 🎯 Training Strategies

### 1. Pure MAE Loss
- **Focus:** Point-wise accuracy
- **Best for:** Applications requiring precise value prediction
- **Training time:** ~2-3 hours per 100 epochs
- **Memory usage:** ~6GB GPU memory

### 2. Pure Autocorrelation Loss  
- **Focus:** Temporal pattern preservation
- **Best for:** Applications requiring realistic dynamics
- **Training time:** ~3-4 hours per 100 epochs
- **Memory usage:** ~8GB GPU memory

### 3. Combined Loss (50% MAE + 50% Autocorr)
- **Focus:** Balanced accuracy and patterns
- **Best for:** General-purpose sequence generation
- **Training time:** ~4-5 hours per 100 epochs
- **Memory usage:** ~10GB GPU memory

---

## 🔮 Model Prediction

Once you have trained models, you can use them for two main prediction tasks:

### 1. Sequence Generation

Generate complete temporal sequences (151 timesteps) from initial conditions:

```bash
cd sequence_generator/

# Edit run_custom_prediction.py to set:
# - input_values: Initial PCA components and parameters
# - model_path: Path to trained model
# - output_dir: Where to save results

python run_custom_prediction.py
```

**Input Configuration:**
```python
# Set initial conditions (7 features)
input_values = np.array([
    1.343466,   # comp_1 (PCA component 1)
    3.098296,   # comp_2 (PCA component 2)  
    -2.72734,   # comp_3 (PCA component 3)
    -1.76372,   # comp_4 (PCA component 4)
    -1.44867,   # comp_5 (PCA component 5)
    0.379756,   # mobility (phase field parameter)
    1.227265    # gradient_coefficient (phase field parameter)
])

# Select trained model
model_path = "D:/neural network/50mae_50auto_401_500/best_model.pth"
```

**Outputs:**
- `prediction.csv`: Complete sequence data (151 timesteps × 8 columns)
- Individual component plots: `comp_1_prediction.png`, etc.

### 2. Time Position Deduction

Find the most likely time positions where given target values occur:

```bash
# Edit simple_time_deduction.py to set:
# - prediction_file: Path to generated prediction.csv
# - target_values: Known values to match

python simple_time_deduction.py
```

**Target Configuration:**
```python
# Set target values (use np.nan for unknown components)
target_values = np.array([
    0.281879,   # comp_1 target (known)
    -2.27557,   # comp_2 target (known)
    -0.53413,   # comp_3 target (known)
    -0.08212,   # comp_4 target (known)
    0.162859,   # comp_5 target (known)
    np.nan,     # mobility (unknown)
    np.nan      # gradient_coefficient (unknown)
])
```

**Outputs:**
- `deduction_results.csv`: Top 10 most likely time positions with probabilities
- `probability_distribution.png`: Probability distribution plot

### Model Selection for Prediction

**Choose model based on your needs:**

```python
# For maximum point-wise accuracy
model_path = "D:/neural network/100mae_401_500/best_model.pth"

# For realistic temporal dynamics  
model_path = "D:/neural network/100auto_401_500/best_model.pth"

# For balanced performance (recommended)
model_path = "D:/neural network/50mae_50auto_401_500/best_model.pth"
```

### Typical Prediction Workflow

1. **Forward Prediction**: Start with initial conditions → generate full sequence
2. **Time Deduction**: Have target state → find when it most likely occurs
3. **Model Validation**: Compare predictions with experimental data
4. **Parameter Estimation**: Use inverse prediction to estimate unknown parameters

For detailed prediction usage, see **[prediction.md](sequence_generator/prediction.md)**.

### Progressive 500-Epoch Strategy

The project implements a sophisticated 5-phase training strategy:

| Phase | Epochs | Learning Rate | Teacher Forcing | Max Lag | Noise Level |
|-------|--------|---------------|-----------------|---------|-------------|
| 1     | 1-100  | 0.001→0.0005  | 0.7→0.6         | 10      | 0.05        |
| 2     | 101-200| 0.0005→0.0001 | 0.6→0.5         | 15      | 0.04        |
| 3     | 201-300| 0.0001→0.00005| 0.5→0.4         | 15      | 0.03        |
| 4     | 301-400| 0.00005→0.00001| 0.4→0.3        | 20      | 0.02        |
| 5     | 401-500| 0.00001→0.000005| 0.3→0.2       | 20      | 0.01        |

## ⚙️ Configuration

### Directory Structure

**Default paths (configurable in scripts):**
- Simulation data: `D:/sample/`
- PCA results: `D:/sample1/`
- Extracted data: `D:/sample2/`
- Training data: `D:/sample3/`
- Model checkpoints: `D:/neural network/`

**To change paths, edit:**
- `sample/phasefield.py` line 98: `base_dir = "your_path"`
- `sample/AutoandPCA.py` line 23: `base_output_path = "your_path"`
- Batch files: `--data_dir` and `--output_dir` parameters

### Hardware Optimization

**For limited GPU memory:**
```bash
# Reduce batch size
--batch_size 16

# Disable mixed precision
# Remove --mixed_precision flag

# Reduce model size
--hidden_size 128 --num_layers 1
```

**For faster training:**
```bash
# Increase batch size (if memory allows)
--batch_size 64

# Use more workers
--num_workers 8

# Enable mixed precision (default)
--mixed_precision
```

### Custom Parameters

**Modify training parameters in batch files:**

```batch
python train_combined_mae_autocorr.py ^
    --data_dir "your_data_path" ^
    --output_dir "your_output_path" ^
    --epochs 100 ^
    --batch_size 32 ^
    --hidden_size 256 ^
    --start_lr 0.001 ^
    --max_lag 10
```

## 📚 Documentation

### Detailed Guides

- **[sample_produce.md](sample/sample_produce.md)**: Complete data generation pipeline
- **[training.md](sequence_generator/training.md)**: Neural network training guide
- **[prediction.md](sequence_generator/prediction.md)**: Model prediction and time deduction guide
- **[realization.md](realization.md)**: Technical implementation details

### Key Concepts

**Phase Field Simulation:**
- Cahn-Hilliard equation for spinodal decomposition
- Randomized parameters for diverse training data
- VTK output format for visualization

**PCA Analysis:**
- Spatial autocorrelation extraction
- Dimensionality reduction to 5 components
- Temporal consistency preservation

**Neural Architecture:**
- LSTM encoder-decoder structure
- Teacher forcing with progressive decay
- Mixed precision training support

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Solution: Reduce batch size
--batch_size 16

# Or disable mixed precision
# Remove --mixed_precision flag
```

**2. Slow Training**
```bash
# Check GPU utilization
nvidia-smi

# Increase batch size if memory allows
--batch_size 64

# Use more data workers
--num_workers 8
```

**3. Poor Convergence**
```bash
# Reduce learning rate
--start_lr 0.0005

# Increase gradient clipping
--clip_grad_norm 2.0

# Adjust teacher forcing
--teacher_forcing_start 0.8
```

**4. File Path Issues**
- Ensure all paths use forward slashes `/` or double backslashes `\\`
- Create directories manually if they don't exist
- Check file permissions

### Performance Monitoring

**Monitor training progress:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training metrics
df = pd.read_csv('D:/neural network/your_model/training_metrics.csv')

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['Epoch'], df['Train Loss'], label='Train')
plt.plot(df['Epoch'], df['Validation Loss'], label='Validation')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(df['Epoch'], df['Learning Rate'])
plt.title('Learning Rate Schedule')
plt.show()
```

### Data Quality Checks

**Verify simulation data:**
```python
import numpy as np
import os

# Check VTK file count
batch_path = "D:/sample/your_batch/"
for folder in os.listdir(batch_path):
    vtk_files = len([f for f in os.listdir(os.path.join(batch_path, folder)) if f.endswith('.vtk')])
    print(f"{folder}: {vtk_files} VTK files")
```

**Verify training data:**
```python
import pandas as pd

# Check CSV data
df = pd.read_csv('D:/sample3/1.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Value ranges:")
print(df.describe())
```

## 🤝 Contributing

### Development Setup

```bash
# Clone for development
git clone <repository-url>
cd phase-field-neural-sequence-generation

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy
```

### Code Style

```bash
# Format code
black .

# Check style
flake8 .

# Type checking
mypy .
```

### Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_data_generation.py
```

### Feature Requests

1. **Create an issue** describing the feature
2. **Fork the repository**
3. **Create a feature branch**: `git checkout -b feature/amazing-feature`
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Phase Field Theory**: Based on Cahn-Hilliard equation for phase separation
- **Neural Architecture**: LSTM encoder-decoder with teacher forcing
- **Training Strategies**: Inspired by sequence-to-sequence learning research

## 📞 Support

**Issues and Questions:**
- Create an issue on GitHub
- Check existing documentation first
- Provide system information and error logs

**Performance Optimization:**
- Share hardware specifications
- Include training logs and metrics
- Describe specific performance goals

**Feature Discussion:**
- Open a discussion thread
- Explain use case and requirements
- Propose implementation approach

---

**Happy Training! 🚀**

For detailed technical information, see [realization.md](realization.md).
For training specifics, see [training.md](sequence_generator/training.md).
For data pipeline details, see [sample_produce.md](sample/sample_produce.md).
