import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlined_model import StreamlinedSequenceGenerator

# Input values - using the values provided
input_values = np.array([1.343466,3.098296,-2.72734,-1.76372,-1.44867,0.379756,1.227265])

# Model path
model_path = "D:/neural network/combined_approach/best_model_combined.pth"

# Output directory - using the path provided
output_dir = "C:/Users/HP/Desktop/graduation project/test/0.37975616521511113,1.2272648537008632,0.6652195021080807/prediction"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print(f"Loading model from {model_path}")
try:
    checkpoint = torch.load(model_path, map_location=device)
    print("Model checkpoint loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Create model
model = StreamlinedSequenceGenerator(
    input_size=7,
    hidden_size=256,
    num_layers=2,
    output_seq_len=151,
    dropout=0.3
).to(device)

# Load state dict
try:
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Model weights loaded successfully")
except Exception as e:
    print(f"Error loading model weights: {e}")
    sys.exit(1)

# Generate prediction
model.eval()
print(f"Input values: {input_values}")

# Prepare input tensor
input_tensor = torch.tensor(input_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
input_tensor = input_tensor.to(device)

# Generate prediction
with torch.no_grad():
    prediction = model(input_tensor)

# Convert to numpy
predicted_sequence = prediction.squeeze(0).cpu().numpy()

# Force the first step to match the input values
predicted_sequence[0] = input_values

# Save prediction to CSV
time_steps = np.arange(151)
df = pd.DataFrame(predicted_sequence, columns=[
    'comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5', 'mobility', 'gradient_coefficient'
])
df.insert(0, 'step', time_steps)

output_file = os.path.join(output_dir, "prediction.csv")
df.to_csv(output_file, index=False)
print(f"Prediction saved to {output_file}")

# Plot each component
component_names = ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5', 'mobility', 'gradient_coefficient']
for i, name in enumerate(component_names):
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, predicted_sequence[:, i], 'r-')
    plt.title(f'{name} - Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f"{name}_prediction.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")

print(f"\nPrediction complete! Results saved to {output_dir}")
