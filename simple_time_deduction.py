import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax

def deduce_time_position(prediction_file, target_values):
    """Simple time position deduction from a prediction file"""
    # Load prediction data
    print(f"Loading prediction from: {prediction_file}")
    df = pd.read_csv(prediction_file)
    print(f"Loaded prediction with {len(df)} steps")
    
    # Extract sequence data
    component_columns = ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5', 'mobility', 'gradient_coefficient']
    sequence_data = df[component_columns].values
    
    # Define component weights (inverse of loss - higher means more important)
    # These are based on the typical loss patterns observed in training
    weights = np.array([
        0.85,  # comp_1
        0.88,  # comp_2
        0.82,  # comp_3
        0.80,  # comp_4
        0.75,  # comp_5
        0.95,  # mobility
        0.95   # gradient_coefficient
    ])
    
    # Handle missing values in target (indicated by NaN)
    valid_mask = ~np.isnan(target_values)
    print(f"Valid components: {np.sum(valid_mask)} out of {len(target_values)}")
    
    # Calculate distances for each time step
    distances = np.zeros(len(df))
    for step in range(len(df)):
        step_values = sequence_data[step]
        
        # Calculate weighted squared difference for valid components
        squared_diff = 0
        for i in range(len(target_values)):
            if valid_mask[i]:
                component_diff = (step_values[i] - target_values[i]) ** 2
                squared_diff += component_diff * weights[i]
        
        distances[step] = squared_diff
    
    # Convert distances to probabilities using softmax
    temperature = 0.1  # Controls how peaked the distribution is
    negative_distances = -distances / temperature
    probabilities = softmax(negative_distances)
    
    # Get top 10 most likely time steps
    top_k = 10
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_probabilities = probabilities[top_indices]
    
    # Print results
    print("\n=== Time Deduction Results ===")
    print(f"Top {top_k} most likely time steps:")
    
    results = []
    for i, (idx, prob) in enumerate(zip(top_indices, top_probabilities)):
        step = df.iloc[idx]["step"]
        step_values = sequence_data[idx]
        
        print(f"Rank {i+1}: Step {int(step)}, Probability: {prob:.6f}")
        
        result = {
            "Rank": i+1,
            "Step": int(step),
            "Probability": prob
        }
        
        # Print component values and differences
        for j, comp in enumerate(component_columns):
            if valid_mask[j]:
                diff = abs(target_values[j] - step_values[j])
                print(f"  {comp}: Target={target_values[j]:.6f}, Sequence={step_values[j]:.6f}, Diff={diff:.6f}")
                
                result[f"{comp}_Target"] = target_values[j]
                result[f"{comp}_Sequence"] = step_values[j]
                result[f"{comp}_Diff"] = diff
            else:
                print(f"  {comp}: Target=N/A, Sequence={step_values[j]:.6f}")
                
                result[f"{comp}_Target"] = np.nan
                result[f"{comp}_Sequence"] = step_values[j]
                result[f"{comp}_Diff"] = np.nan
        
        results.append(result)
    
    # Save results to CSV
    output_dir = os.path.dirname(prediction_file)
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "deduction_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Plot probability distribution
    plt.figure(figsize=(12, 6))
    steps = df.iloc[top_indices]["step"].values
    plt.bar(steps, top_probabilities, color='skyblue')
    plt.xlabel('Time Step')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of Most Likely Time Steps')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "probability_distribution.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")
    
    return top_indices, top_probabilities

if __name__ == "__main__":
    # Prediction file path
    prediction_file = "C:/Users/HP/Desktop/graduation project/test/0.4595638295185252,1.9376728706862238,0.8797164603925978/prediction/prediction.csv"
    
    # Target values with missing components (NaN)
    # Format: [comp_1, comp_2, comp_3, comp_4, comp_5, mobility, gradient_coefficient]
    target_values = np.array([0.281879,-2.27557,-0.53413,-0.08212,0.162859, np.nan, np.nan])
    
    # Run time deduction
    deduce_time_position(prediction_file, target_values)
