# Model Prediction and Time Deduction Documentation

This document explains how to use trained models for sequence prediction and time position deduction. The workflow consists of two main components: generating predictions from trained models and finding the most likely time positions for given target values.

## Overview

The prediction system provides two key functionalities:

1. **Sequence Generation** (`run_custom_prediction.py`): Generate complete 151-step sequences from initial conditions
2. **Time Deduction** (`simple_time_deduction.py`): Find most likely time positions for given target values

## Table of Contents

- [Sequence Generation](#sequence-generation)
- [Time Position Deduction](#time-position-deduction)
- [Complete Workflow](#complete-workflow)
- [Configuration Guide](#configuration-guide)
- [Output Analysis](#output-analysis)
- [Troubleshooting](#troubleshooting)

---

## Sequence Generation

### Purpose

Generate a complete temporal sequence (151 timesteps) from initial conditions using a trained neural network model.

### Script: `run_custom_prediction.py`

#### Key Components

**1. Input Configuration:**
```python
# Initial condition values (7 features)
input_values = np.array([
    1.343466,   # comp_1 (PCA component 1)
    3.098296,   # comp_2 (PCA component 2)
    -2.72734,   # comp_3 (PCA component 3)
    -1.76372,   # comp_4 (PCA component 4)
    -1.44867,   # comp_5 (PCA component 5)
    0.379756,   # mobility (phase field parameter)
    1.227265    # gradient_coefficient (phase field parameter)
])
```

**2. Model Configuration:**
```python
# Path to trained model
model_path = "D:/neural network/combined_approach/best_model_combined.pth"

# Model architecture (must match training)
model = StreamlinedSequenceGenerator(
    input_size=7,           # 7 input features
    hidden_size=256,        # Hidden layer size
    num_layers=2,           # Number of LSTM layers
    output_seq_len=151,     # Output sequence length
    dropout=0.3             # Dropout rate
)
```

**3. Output Configuration:**
```python
# Output directory for results
output_dir = "C:/Users/HP/Desktop/graduation project/test/0.37975616521511113,1.2272648537008632,0.6652195021080807/prediction"
```

### How to Use

#### Step 1: Prepare Input Values

**Option A: From Known Phase Field Parameters**
```python
# If you know the initial phase field parameters
input_values = np.array([
    initial_comp_1,     # From PCA analysis of initial field
    initial_comp_2,     # From PCA analysis of initial field
    initial_comp_3,     # From PCA analysis of initial field
    initial_comp_4,     # From PCA analysis of initial field
    initial_comp_5,     # From PCA analysis of initial field
    mobility,           # Known mobility parameter
    gradient_coef       # Known gradient coefficient
])
```

**Option B: From Experimental Data**
```python
# If you have PCA components from experimental observations
input_values = np.array([
    measured_comp_1,    # From experimental PCA analysis
    measured_comp_2,    # From experimental PCA analysis
    measured_comp_3,    # From experimental PCA analysis
    measured_comp_4,    # From experimental PCA analysis
    measured_comp_5,    # From experimental PCA analysis
    estimated_mobility, # Estimated or fitted parameter
    estimated_grad_coef # Estimated or fitted parameter
])
```

#### Step 2: Select Trained Model

**Available Model Types:**
```python
# Pure MAE trained model
model_path = "D:/neural network/100mae_401_500/best_model.pth"

# Pure Autocorrelation trained model
model_path = "D:/neural network/100auto_401_500/best_model.pth"

# Combined approach trained model (recommended)
model_path = "D:/neural network/50mae_50auto_401_500/best_model.pth"
```

**Model Selection Criteria:**
- **Combined approach**: Best overall performance, recommended for most use cases
- **Pure MAE**: When point-wise accuracy is most important
- **Pure Autocorrelation**: When temporal pattern preservation is critical

#### Step 3: Configure Output Directory

```python
# Create meaningful output directory names
output_dir = f"predictions/{mobility}_{gradient_coef}_{timestamp}"
# OR use parameter-based naming
output_dir = f"C:/Users/HP/Desktop/results/{c0},{mobility},{grad_coef}/prediction"
```

#### Step 4: Run Prediction

```bash
cd sequence_generator/
python run_custom_prediction.py
```

### Output Files

**Generated Files:**
1. **`prediction.csv`**: Complete sequence data (151 rows × 8 columns)
2. **Individual plots**: `comp_1_prediction.png`, `comp_2_prediction.png`, etc.

**CSV Structure:**
```csv
step,comp_1,comp_2,comp_3,comp_4,comp_5,mobility,gradient_coefficient
0,1.343466,3.098296,-2.72734,-1.76372,-1.44867,0.379756,1.227265
1,1.341234,3.095123,-2.72156,-1.76089,-1.44523,0.379756,1.227265
...
150,0.987654,2.123456,-1.98765,-1.23456,-0.98765,0.379756,1.227265
```

---

## Time Position Deduction

### Purpose

Find the most likely time positions in a predicted sequence that match given target values, useful for:
- Validation against experimental data
- Time point identification in sequences
- Model accuracy assessment

### Script: `simple_time_deduction.py`

#### Key Components

**1. Prediction File Input:**
```python
# Path to generated prediction file
prediction_file = "C:/Users/HP/Desktop/graduation project/test/0.4595638295185252,1.9376728706862238,0.8797164603925978/prediction/prediction.csv"
```

**2. Target Values:**
```python
# Target values to match (NaN for unknown/missing values)
target_values = np.array([
    0.281879,   # comp_1 target
    -2.27557,   # comp_2 target
    -0.53413,   # comp_3 target
    -0.08212,   # comp_4 target
    0.162859,   # comp_5 target
    np.nan,     # mobility (unknown)
    np.nan      # gradient_coefficient (unknown)
])
```

**3. Component Weights:**
```python
# Importance weights for each component (based on training loss patterns)
weights = np.array([
    0.85,  # comp_1 weight
    0.88,  # comp_2 weight
    0.82,  # comp_3 weight
    0.80,  # comp_4 weight
    0.75,  # comp_5 weight
    0.95,  # mobility weight
    0.95   # gradient_coefficient weight
])
```

### How to Use

#### Step 1: Obtain Target Values

**Option A: From Experimental Measurements**
```python
# PCA analysis of experimental microstructure at known time
experimental_field = load_experimental_data("experiment_t500.tif")
autocorr = calculate_autocorrelation(experimental_field)
pca_components = pca_model.transform(autocorr.flatten().reshape(1, -1))

target_values = np.array([
    pca_components[0, 0],  # comp_1
    pca_components[0, 1],  # comp_2
    pca_components[0, 2],  # comp_3
    pca_components[0, 3],  # comp_4
    pca_components[0, 4],  # comp_5
    np.nan,                # mobility (unknown)
    np.nan                 # gradient_coefficient (unknown)
])
```

**Option B: From Known Simulation Data**
```python
# Values from a reference simulation at specific timestep
reference_data = pd.read_csv("reference_simulation.csv")
target_timestep = 100  # Known timestep

target_values = reference_data.iloc[target_timestep][
    ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5', 'mobility', 'gradient_coefficient']
].values
```

**Option C: Partial Information**
```python
# When only some components are known
target_values = np.array([
    known_comp_1,   # From measurement
    known_comp_2,   # From measurement
    np.nan,         # comp_3 unknown
    np.nan,         # comp_4 unknown
    np.nan,         # comp_5 unknown
    known_mobility, # From experimental conditions
    np.nan          # gradient_coefficient unknown
])
```

#### Step 2: Configure Prediction File Path

```python
# Path to the prediction file generated by run_custom_prediction.py
prediction_file = "path/to/your/prediction/prediction.csv"
```

#### Step 3: Adjust Component Weights (Optional)

```python
# Modify weights based on confidence in measurements
weights = np.array([
    0.90,  # comp_1 - high confidence measurement
    0.85,  # comp_2 - medium confidence  
    0.70,  # comp_3 - lower confidence
    0.75,  # comp_4 - medium confidence
    0.80,  # comp_5 - good confidence
    0.95,  # mobility - known parameter
    0.95   # gradient_coefficient - known parameter
])
```

#### Step 4: Run Time Deduction

```bash
cd sequence_generator/
python simple_time_deduction.py
```

### Output Files

**Generated Files:**
1. **`deduction_results.csv`**: Top 10 most likely time positions with detailed analysis
2. **`probability_distribution.png`**: Bar chart of probability distribution

**Results CSV Structure:**
```csv
Rank,Step,Probability,comp_1_Target,comp_1_Sequence,comp_1_Diff,...
1,45,0.234567,0.281879,0.278234,0.003645,...
2,47,0.198234,0.281879,0.285123,0.003244,...
...
```

---

## Complete Workflow

### Typical Use Case: Model Validation

#### Step 1: Generate Prediction
```python
# 1. Set initial conditions from experimental setup
input_values = np.array([initial_pca_components, mobility, grad_coef])

# 2. Select appropriate trained model
model_path = "D:/neural network/50mae_50auto_401_500/best_model.pth"

# 3. Run prediction
python run_custom_prediction.py
```

#### Step 2: Compare with Experimental Data
```python
# 1. Obtain target values from experimental measurement at known time
target_values = experimental_pca_at_t300

# 2. Set prediction file path
prediction_file = "output/prediction.csv"

# 3. Run time deduction
python simple_time_deduction.py
```

#### Step 3: Analyze Results
```python
# Check if predicted time matches experimental time
results = pd.read_csv("output/deduction_results.csv")
predicted_time = results.iloc[0]["Step"]  # Top prediction
actual_time = 300  # Known experimental time

accuracy = abs(predicted_time - actual_time) / actual_time * 100
print(f"Time prediction accuracy: {100-accuracy:.2f}%")
```

### Forward Prediction Workflow

```python
# 1. Start with known initial conditions
initial_conditions = [comp1, comp2, comp3, comp4, comp5, mobility, grad_coef]

# 2. Generate full sequence
python run_custom_prediction.py

# 3. Extract prediction at desired time
prediction_df = pd.read_csv("prediction.csv")
future_state = prediction_df.iloc[target_timestep]

print(f"Predicted state at timestep {target_timestep}:")
print(future_state)
```

### Inverse Problem Workflow

```python
# 1. Have target state, want to find when it occurs
target_state = [known_comp1, known_comp2, ..., known_params]

# 2. Generate sequence from estimated initial conditions
python run_custom_prediction.py

# 3. Find when target state most likely occurs
python simple_time_deduction.py

# 4. Refine initial conditions based on results
```

---

## Configuration Guide

### Model Selection

**Choose model based on application:**

```python
# For maximum accuracy (point-wise)
model_path = "D:/neural network/100mae_401_500/best_model.pth"

# For realistic temporal dynamics
model_path = "D:/neural network/100auto_401_500/best_model.pth"  

# For balanced performance (recommended)
model_path = "D:/neural network/50mae_50auto_401_500/best_model.pth"
```

### Input Value Ranges

**Typical value ranges for each component:**

```python
# PCA Components (typical ranges from training data)
comp_1_range = [-3.0, 3.0]    # Primary spatial pattern
comp_2_range = [-2.5, 2.5]    # Secondary spatial pattern  
comp_3_range = [-2.0, 2.0]    # Tertiary spatial pattern
comp_4_range = [-1.5, 1.5]    # Quaternary spatial pattern
comp_5_range = [-1.0, 1.0]    # Fifth spatial pattern

# Phase Field Parameters
mobility_range = [1.0, 2.0]           # Diffusion rate
grad_coef_range = [0.5, 1.0]          # Interface energy
```

**Input validation:**
```python
def validate_inputs(input_values):
    """Validate input values are within reasonable ranges"""
    ranges = [[-3, 3], [-2.5, 2.5], [-2, 2], [-1.5, 1.5], [-1, 1], [1, 2], [0.5, 1]]
    
    for i, (val, (min_val, max_val)) in enumerate(zip(input_values, ranges)):
        if not (min_val <= val <= max_val):
            print(f"Warning: Input {i} ({val}) outside typical range [{min_val}, {max_val}]")
```

### Temperature Parameter Tuning

**Adjust probability sharpness in time deduction:**

```python
# In simple_time_deduction.py
temperature = 0.1   # Sharp distribution (confident predictions)
temperature = 0.5   # Moderate distribution  
temperature = 1.0   # Broader distribution (less confident)
```

**Effect of temperature:**
- **Low temperature (0.1)**: Sharp peaks, confident predictions
- **High temperature (1.0)**: Broader distribution, less confident
- **Very high temperature (10.0)**: Nearly uniform distribution

---

## Output Analysis

### Interpreting Prediction Results

**Sequence Evolution Patterns:**

```python
# Load and analyze prediction
df = pd.read_csv("prediction.csv")

# Check for realistic evolution
for col in ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5']:
    initial_val = df[col].iloc[0]
    final_val = df[col].iloc[-1]
    total_change = abs(final_val - initial_val)
    
    print(f"{col}: {initial_val:.3f} → {final_val:.3f} (change: {total_change:.3f})")
```

**Temporal Smoothness Check:**
```python
# Check for unrealistic jumps
for col in ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5']:
    differences = np.diff(df[col])
    max_jump = np.max(np.abs(differences))
    
    if max_jump > 0.1:  # Threshold for suspicious jumps
        print(f"Warning: Large jump detected in {col}: {max_jump:.3f}")
```

### Interpreting Time Deduction Results

**Confidence Assessment:**
```python
results = pd.read_csv("deduction_results.csv")

# Check top prediction confidence
top_prob = results.iloc[0]["Probability"]
second_prob = results.iloc[1]["Probability"]
confidence_ratio = top_prob / second_prob

if confidence_ratio > 2.0:
    print("High confidence prediction")
elif confidence_ratio > 1.5:
    print("Moderate confidence prediction")
else:
    print("Low confidence prediction - multiple likely candidates")
```

**Component Contribution Analysis:**
```python
# Analyze which components drive the prediction
for i in range(5):  # Check each component
    diff_col = f"comp_{i+1}_Diff"
    if diff_col in results.columns:
        avg_diff = results[diff_col].mean()
        print(f"Average difference for comp_{i+1}: {avg_diff:.6f}")
```

---

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```python
# Error: Model architecture mismatch
# Solution: Verify model parameters match training configuration
model = StreamlinedSequenceGenerator(
    input_size=7,      # Must match training
    hidden_size=256,   # Must match training  
    num_layers=2,      # Must match training
    output_seq_len=151,# Must match training
    dropout=0.3        # Must match training
)
```

**2. Input Value Errors**
```python
# Error: Input values out of range
# Solution: Check and normalize input values
input_values = np.clip(input_values, [-3, -2.5, -2, -1.5, -1, 1, 0.5], 
                                     [3, 2.5, 2, 1.5, 1, 2, 1])
```

**3. File Path Issues**
```python
# Error: File not found
# Solution: Use absolute paths and check existence
import os

if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    
if not os.path.exists(prediction_file):
    print(f"Prediction file not found: {prediction_file}")
```

**4. Memory Issues**
```python
# Error: CUDA out of memory
# Solution: Use CPU or reduce model size
device = torch.device("cpu")  # Force CPU usage

# Or check GPU memory
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Validation Checks

**Model Sanity Check:**
```python
def validate_prediction(prediction_array):
    """Check if prediction is reasonable"""
    # Check for NaN values
    if np.any(np.isnan(prediction_array)):
        print("Warning: NaN values in prediction")
    
    # Check for extreme values
    if np.any(np.abs(prediction_array[:, :5]) > 5):  # PCA components
        print("Warning: Extreme PCA component values")
    
    # Check parameter consistency
    mobility_std = np.std(prediction_array[:, 5])
    grad_coef_std = np.std(prediction_array[:, 6])
    
    if mobility_std > 0.01:  # Should be constant
        print("Warning: Mobility parameter varies in prediction")
    if grad_coef_std > 0.01:  # Should be constant
        print("Warning: Gradient coefficient varies in prediction")
```

**Time Deduction Validation:**
```python
def validate_deduction_results(results_df):
    """Check if time deduction results are reasonable"""
    # Check probability sum
    prob_sum = results_df["Probability"].sum()
    if abs(prob_sum - 1.0) > 0.01:
        print(f"Warning: Probabilities don't sum to 1.0 (sum: {prob_sum:.3f})")
    
    # Check for reasonable time spread
    time_spread = results_df["Step"].max() - results_df["Step"].min()
    if time_spread < 5:
        print("Warning: All predictions clustered in narrow time range")
    elif time_spread > 100:
        print("Warning: Predictions spread over very wide time range")
```

### Performance Optimization

**Faster Prediction:**
```python
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable JIT compilation for repeated predictions
model = torch.jit.script(model)

# Batch multiple predictions
input_batch = torch.stack([input_tensor1, input_tensor2, input_tensor3])
predictions = model(input_batch)
```

**Memory Optimization:**
```python
# Clear GPU cache
torch.cuda.empty_cache()

# Use CPU for very large sequences
device = torch.device("cpu")

# Reduce precision if needed
model = model.half()  # Use FP16
```

This comprehensive prediction documentation provides users with everything needed to effectively use the trained models for both forward prediction and inverse time deduction tasks.
