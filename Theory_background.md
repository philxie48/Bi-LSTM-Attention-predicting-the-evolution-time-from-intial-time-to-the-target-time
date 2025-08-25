# Theoretical Background

This document provides the theoretical foundations underlying the spinodal decomposition simulation, principal component analysis, and neural network architectures implemented in this project.

## Table of Contents

1. [Spinodal Decomposition Theory](#spinodal-decomposition-theory)
2. [Principal Component Analysis Theory](#principal-component-analysis-theory)
3. [Neural Network Architecture Theory](#neural-network-architecture-theory)
4. [Loss Function Theory](#loss-function-theory)
5. [Training Algorithm Theory](#training-algorithm-theory)

---

## Spinodal Decomposition Theory

### Physical Background

Spinodal decomposition is a mechanism of phase separation in binary alloys that occurs when a homogeneous mixture becomes thermodynamically unstable. Unlike nucleation and growth, spinodal decomposition proceeds through spontaneous fluctuations in composition that grow exponentially with time.

### Thermodynamic Foundation

#### Free Energy Functional

The total free energy of a binary system is described by the Ginzburg-Landau functional:

```
F[c] = ∫∫ [f(c) + (κ/2)|∇c|²] dV
```

Where:
- **f(c)**: Bulk free energy density (chemical energy)
- **κ**: Gradient energy coefficient (interface energy)
- **c(x,y,t)**: Local concentration field
- **∇c**: Concentration gradient

#### Double-Well Potential

The bulk free energy density is modeled as a double-well potential:

```
f(c) = c²(1-c)²
```

**Mathematical Properties:**
- **Minima at c = 0 and c = 1**: Pure phases are energetically favorable
- **Maximum at c = 0.5**: Mixed state is unstable
- **Concave region (0.211 < c < 0.789)**: Thermodynamically unstable (spinodal region)

**Physical Interpretation:**
- The double-well shape represents the tendency for phase separation
- The barrier height controls the difficulty of phase mixing
- The well curvature determines the driving force for decomposition

### Cahn-Hilliard Equation

#### Derivation from Thermodynamics

The evolution of the concentration field is governed by the principle of mass conservation and the minimization of free energy:

```
∂c/∂t = ∇ · (M(c) ∇μ)
```

Where the chemical potential is:

```
μ = δF/δc = ∂f/∂c - κ∇²c
```

For constant mobility M, this simplifies to:

```
∂c/∂t = M ∇²μ = M ∇²(f'(c) - κ∇²c)
```

#### Mathematical Structure

**Fourth-Order PDE:**
- The Cahn-Hilliard equation is a fourth-order nonlinear parabolic PDE
- Combines second-order diffusion (∇²) with fourth-order surface tension effects (∇⁴)
- Conserves total mass: ∫ c dV = constant

**Free Energy Derivative:**
```
f'(c) = df/dc = 2c(1-c)² - 2c²(1-c) = 2c(1-c)(1-2c)
```

**Physical Meaning:**
- **c < 0.5**: f'(c) < 0, driving concentration toward c = 0
- **c > 0.5**: f'(c) > 0, driving concentration toward c = 1
- **c = 0.5**: f'(c) = 0, unstable equilibrium

### Numerical Implementation

#### Finite Difference Discretization

**Spatial Discretization (Laplacian):**
```
∇²u_{i,j} ≈ (u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}) / h²
```

**Temporal Discretization (Explicit Euler):**
```
c^{n+1} = c^n + Δt · M · ∇²μ^n
```

#### Stability Considerations

**CFL Condition:**
For stability of the explicit scheme:
```
Δt ≤ h⁴ / (2M κ d)
```
Where d is the spatial dimension.

**Numerical Parameters in Implementation:**
- **Grid size**: 128×128 (h = 1.0)
- **Time step**: Δt = 0.001
- **Mobility**: M ∈ [1, 2]
- **Gradient coefficient**: κ ∈ [0.5, 1.0]

### Energy Evolution

#### Total Energy Calculation

The discrete form of total energy:
```
E = Σᵢⱼ [f(cᵢⱼ) + (κ/2)(|∇c|²)ᵢⱼ] h²
```

**Properties:**
- Energy decreases monotonically: dE/dt ≤ 0
- Equilibrium reached when dE/dt = 0
- Rate of decrease proportional to mobility M

#### Phase Separation Dynamics

**Early Stage (Linear Growth):**
- Small perturbations grow exponentially
- Growth rate depends on wavelength
- Fastest growing mode determines characteristic length scale

**Late Stage (Coarsening):**
- Domain growth follows power law: L(t) ∝ t^α
- Scaling exponent α ≈ 1/3 for diffusion-limited growth
- Surface tension minimization drives coarsening

---

## Principal Component Analysis Theory

### Mathematical Foundation

#### Eigenvalue Decomposition

Given a data matrix X (n×p), PCA finds the directions of maximum variance through eigenvalue decomposition of the covariance matrix:

```
C = (1/(n-1)) X^T X
```

The eigenvalue problem:
```
C vᵢ = λᵢ vᵢ
```

Where:
- **λᵢ**: Eigenvalues (variance along principal component i)
- **vᵢ**: Eigenvectors (principal component directions)
- **λ₁ ≥ λ₂ ≥ ... ≥ λₚ**: Ordered by decreasing variance

#### Dimensionality Reduction

**Projection onto Principal Components:**
```
Y = X V_k
```

Where V_k contains the first k eigenvectors.

**Variance Preservation:**
The fraction of total variance preserved:
```
R_k = (Σᵢ₌₁ᵏ λᵢ) / (Σᵢ₌₁ᵖ λᵢ)
```

### Application to Phase Field Data

#### Spatial Autocorrelation Preprocessing

**2D Autocorrelation Function:**
The spatial autocorrelation is computed using the Wiener-Khintchine theorem:
```
R(τ) = F⁻¹{|F{f(x)}|²}
```

Where F denotes the 2D Fourier transform.

**Mathematical Properties:**
- **R(0) = 1**: Perfect correlation at zero lag
- **R(τ) → 0** as τ → ∞: Decorrelation at large distances
- **Symmetric**: R(τ) = R(-τ)

**Radial Averaging:**
Convert 2D autocorrelation to 1D radial profile:
```
R(r) = ⟨R(τ)⟩_{|τ|=r}
```

#### Feature Extraction Strategy

**Data Preparation:**
1. **Normalization**: Zero mean, unit variance for each timestep
2. **Flattening**: Convert 2D autocorrelation to 1D vector
3. **Concatenation**: Stack all timesteps into matrix X

**PCA Application:**
- **Input**: X (151 timesteps × ~16384 spatial points)
- **Output**: Y (151 timesteps × 5 components)
- **Compression ratio**: ~3277:1

**Component Interpretation:**
- **PC1**: Dominant spatial pattern (highest variance)
- **PC2**: Secondary spatial pattern (orthogonal to PC1)
- **PC3-PC5**: Higher-order spatial modes
- **Temporal evolution**: How each component changes over time

### Temporal Consistency

#### Ordering Preservation

**Critical Requirement:**
PCA must preserve temporal ordering of data:
```
t₁ < t₂ ⟹ index(t₁) < index(t₂)
```

**Implementation:**
1. Sort timesteps before PCA
2. Maintain index mapping
3. Verify monotonic ordering

**Physical Justification:**
- Neural networks expect temporally consistent sequences
- Causality must be preserved in training data
- Smooth evolution ensures realistic dynamics

---

## Neural Network Architecture Theory

### Sequence-to-Sequence Learning

#### Problem Formulation

**Input Sequence**: X = {x₁, x₂, ..., x_T}
**Output Sequence**: Y = {y₁, y₂, ..., y_T}
**Goal**: Learn mapping f: X → Y

**Challenges:**
- Variable length sequences
- Long-term dependencies
- Temporal correlation preservation

#### Encoder-Decoder Architecture

**Encoder**: Compress input sequence to fixed-size representation
```
h_t = LSTM_enc(x_t, h_{t-1})
c = g(h_T)  // Context vector
```

**Decoder**: Generate output sequence from context
```
s_t = LSTM_dec(y_{t-1}, s_{t-1}, c)
y_t = softmax(W_s s_t + b_s)
```

### LSTM (Long Short-Term Memory)

#### Cell State Dynamics

The LSTM addresses the vanishing gradient problem through gated cell states:

**Forget Gate:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**Input Gate:**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

**Cell State Update:**
```
C_t = f_t * C_{t-1} + i_t * C̃_t
```

**Output Gate:**
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

#### Mathematical Properties

**Gradient Flow:**
The cell state provides a highway for gradient flow:
```
∂C_t/∂C_{t-1} = f_t
```

**Long-term Memory:**
When f_t ≈ 1, gradients can flow unchanged across many timesteps.

**Selective Memory:**
Gates allow the network to selectively remember, forget, and update information.

### Teacher Forcing

#### Training Strategy

**Standard Approach:**
Use ground truth as input during training:
```
ŷ_t = LSTM(y_{t-1}, h_{t-1})  // y_{t-1} is ground truth
```

**Inference:**
Use model predictions as input:
```
ŷ_t = LSTM(ŷ_{t-1}, h_{t-1})  // ŷ_{t-1} is model prediction
```

#### Exposure Bias Problem

**Issue**: Training-inference mismatch leads to error accumulation
**Solution**: Gradually reduce teacher forcing ratio

**Scheduled Sampling:**
```
p_t = max(ε, k - c·t)  // Linear decay
use_teacher = random() < p_t
input_t = y_{t-1} if use_teacher else ŷ_{t-1}
```

**Implementation in Project:**
- Start: 70% teacher forcing
- End: 20% teacher forcing
- Linear decay over training epochs

### Model Architecture Details

#### StreamlinedSequenceGenerator

**Input Specification:**
- **Sequence length**: 151 timesteps
- **Feature dimension**: 7 (5 PCA + 2 parameters)
- **Batch processing**: 32 sequences

**Layer Configuration:**
```python
# Encoder LSTM
encoder = LSTM(input_size=7, hidden_size=256, num_layers=2, dropout=0.3)

# Decoder LSTM  
decoder = LSTM(input_size=7, hidden_size=256, num_layers=2, dropout=0.3)

# Output projection
output_layer = Linear(hidden_size=256, output_size=7)
```

**Parameter Count:**
- **LSTM parameters**: ~2M parameters
- **Linear layer**: ~1.8K parameters
- **Total**: ~2.1M parameters

#### Information Flow

**Forward Pass:**
1. **Encoder processes input sequence**: X → H (hidden representations)
2. **Context extraction**: H_T → C (context vector)
3. **Decoder generates output**: C → Y (output sequence)

**Hidden State Transfer:**
```python
encoder_output, (h_n, c_n) = encoder(input_sequence)
decoder_output, _ = decoder(decoder_input, (h_n, c_n))
```

---

## Loss Function Theory

### Mean Absolute Error (MAE)

#### Mathematical Definition

```
L_MAE = (1/n) Σᵢ |yᵢ - ŷᵢ|
```

**Properties:**
- **Robust to outliers**: L1 norm less sensitive than L2
- **Linear gradient**: ∂L/∂ŷ = sign(ŷ - y)
- **Point-wise accuracy**: Optimizes absolute prediction error

#### Gradient Analysis

**Gradient magnitude**: |∂L_MAE/∂ŷ| = 1 (constant)
**Convergence**: Slower than MSE but more stable
**Geometric interpretation**: Minimizes sum of absolute deviations

### Autocorrelation Loss

#### Theoretical Foundation

**Temporal Autocorrelation:**
```
R_y(τ) = E[y(t)y(t+τ)] / E[y²(t)]
```

**Loss Function:**
```
L_autocorr = ||R_pred(τ) - R_true(τ)||²
```

#### Implementation Details

**Discrete Autocorrelation:**
```python
def autocorr(x, max_lag):
    result = []
    for lag in range(max_lag + 1):
        if lag == 0:
            result.append(1.0)  # Perfect correlation
        else:
            x1, x2 = x[:-lag], x[lag:]
            corr = torch.sum(x1 * x2) / torch.sum(x1 * x1)
            result.append(corr)
    return torch.stack(result)
```

**Gradient Properties:**
- **Non-local**: Each point affects correlation at multiple lags
- **Temporal coupling**: Enforces consistency across time
- **Scale invariant**: Normalized correlation is amplitude-independent

#### Physical Interpretation

**Pattern Preservation:**
- Maintains characteristic length scales
- Preserves periodic behaviors
- Enforces realistic temporal dynamics

**Mathematical Constraints:**
- **Causality**: R(τ) depends on past and present values
- **Symmetry**: R(τ) = R(-τ) for stationary processes
- **Bounded**: |R(τ)| ≤ R(0) = 1

### Combined Loss Function

#### Weighted Combination

```
L_combined = α L_MAE + (1-α) L_autocorr
```

**Balance Considerations:**
- **α = 0.5**: Equal weighting (used in project)
- **α > 0.5**: Emphasis on point-wise accuracy
- **α < 0.5**: Emphasis on temporal patterns

#### Multi-Objective Optimization

**Pareto Optimality:**
The combined loss seeks solutions on the Pareto frontier between accuracy and pattern preservation.

**Gradient Analysis:**
```
∇L_combined = α ∇L_MAE + (1-α) ∇L_autocorr
```

**Convergence Properties:**
- **Early training**: MAE dominates (large point-wise errors)
- **Late training**: Autocorr refines temporal patterns
- **Equilibrium**: Balance between objectives

---

## Training Algorithm Theory

### Optimization Algorithm

#### AdamW Optimizer

**Parameter Updates:**
```
m_t = β₁ m_{t-1} + (1-β₁) g_t
v_t = β₂ v_{t-1} + (1-β₂) g_t²
θ_{t+1} = θ_t - η (m̂_t/(√v̂_t + ε) + λθ_t)
```

Where:
- **m_t, v_t**: Biased first and second moment estimates
- **m̂_t, v̂_t**: Bias-corrected estimates
- **λ**: Weight decay coefficient
- **η**: Learning rate

**Advantages:**
- **Adaptive learning rates**: Different rates for each parameter
- **Momentum**: Accelerated convergence
- **Weight decay**: L2 regularization for generalization

### Learning Rate Scheduling

#### Warmup + Cosine Decay

**Warmup Phase (30% of training):**
```
lr_t = lr_start + (lr_max - lr_start) × (t / t_warmup)
```

**Cosine Decay Phase (70% of training):**
```
lr_t = lr_min + 0.5(lr_max - lr_min)(1 + cos(π × progress))
```

**Theoretical Justification:**
- **Warmup**: Prevents early instability from large gradients
- **Cosine decay**: Smooth reduction for fine-tuning
- **Annealing**: Improves convergence to local minima

#### Progressive Training Strategy

**Multi-Phase Approach:**
Each phase reduces:
- Learning rate (÷10)
- Teacher forcing ratio (-0.1)
- Input noise level (-0.01)
- Weight decay (÷1.25)

**Mathematical Rationale:**
- **Coarse-to-fine**: Large steps initially, small refinements later
- **Curriculum learning**: Gradually increase task difficulty
- **Regularization annealing**: Reduce constraints as model improves

### Regularization Techniques

#### Gradient Clipping

**Norm Clipping:**
```
if ||g|| > τ:
    g ← τ × g/||g||
```

**Purpose**: Prevent exploding gradients in RNNs
**Effect**: Maintains gradient direction while limiting magnitude

#### Dropout

**Mathematical Model:**
```
y = x ⊙ m / p
```
Where m ~ Bernoulli(p)

**Theoretical Effect:**
- **Training**: Randomly zeros neurons with probability (1-p)
- **Inference**: Scales outputs by 1/p
- **Regularization**: Prevents co-adaptation of neurons

#### Input Noise Injection

**Gaussian Noise:**
```
x̃ = x + σε, where ε ~ N(0,1)
```

**Benefits:**
- **Robustness**: Improves generalization to noisy inputs
- **Smoothing**: Regularizes the loss landscape
- **Data augmentation**: Implicit expansion of training set

### Mixed Precision Training

#### FP16 Computation

**Forward Pass**: Compute in FP16
**Gradient Scaling**: Prevent underflow
**Parameter Update**: Maintain FP32 master weights

**Mathematical Considerations:**
- **Dynamic range**: FP16 has limited range [6×10⁻⁸, 6×10⁴]
- **Precision**: 3-4 significant digits vs 7 for FP32
- **Scaling factor**: Typically 2¹⁶ = 65536

**Benefits:**
- **Memory reduction**: ~50% GPU memory savings
- **Speed improvement**: ~1.5-2× faster on modern GPUs
- **Maintained accuracy**: Careful scaling preserves convergence

### Convergence Analysis

#### Loss Landscape

**MAE Loss Landscape:**
- **Convex locally**: Linear gradients near minima
- **Non-convex globally**: Multiple local minima possible
- **Smooth**: Continuous gradients throughout

**Autocorrelation Loss Landscape:**
- **Highly non-convex**: Complex interaction terms
- **Multi-modal**: Multiple valid temporal patterns
- **Sensitive**: Small changes can affect long-range correlations

**Combined Landscape:**
- **Regularized**: MAE component smooths autocorr landscape
- **Constrained**: Autocorr component restricts MAE solutions
- **Balanced**: Trade-off between competing objectives

#### Training Dynamics

**Phase Transitions:**
1. **Initialization**: Random weights, high loss
2. **Fast learning**: Rapid loss reduction (epochs 1-20)
3. **Refinement**: Slow improvements (epochs 20-80)
4. **Convergence**: Minimal changes (epochs 80-100)

**Loss Component Evolution:**
- **Early**: MAE dominates total loss
- **Middle**: Components become balanced
- **Late**: Fine-tuning of autocorr patterns

This theoretical foundation provides the mathematical and physical basis for understanding the complete pipeline from phase field simulation through neural network training.
