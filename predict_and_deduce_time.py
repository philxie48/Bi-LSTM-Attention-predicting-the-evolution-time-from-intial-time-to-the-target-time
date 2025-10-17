"""
Integrated Prediction and Time Deduction Interface

This script combines sequence prediction and time deduction into a single workflow.
Users can:
1. Select a trained model
2. Input initial component values
3. Generate a full sequence prediction
4. Input target component values
5. Get the most likely time step for the target
"""

import os
import sys

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from streamlined_model import StreamlinedSequenceGenerator
from datetime import datetime


class PredictionAndDeductionPipeline:
    """Main pipeline for prediction and time deduction"""
    
    # Available models
    MODELS = {
        "1": {
            "name": "50% MAE + 50% Autocorr (Epochs 401-500)",
            "path": "D:/neural network/50mae_50auto_401_500/best_model_combined.pth"
        },
        "2": {
            "name": "100% MAE (Epochs 401-500)",
            "path": "D:/neural network/100mae_401_500/best_model_combined.pth"
        }
    }
    
    # Component names
    COMPONENT_NAMES = ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5', 'mobility', 'gradient_coefficient']
    
    # Component weights for time deduction (based on typical loss patterns)
    COMPONENT_WEIGHTS = np.array([0.85, 0.88, 0.82, 0.80, 0.75, 0.95, 0.95])
    
    def __init__(self, model_choice, output_base_dir="D:/results/predictions"):
        """
        Initialize the pipeline.
        
        Args:
            model_choice (str): Model selection ("1" or "2")
            output_base_dir (str): Base directory for saving outputs
        """
        self.model_choice = model_choice
        self.output_base_dir = output_base_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.prediction = None
        self.output_dir = None
        
        print(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the selected model"""
        if self.model_choice not in self.MODELS:
            raise ValueError(f"Invalid model choice. Choose from: {list(self.MODELS.keys())}")
        
        model_info = self.MODELS[self.model_choice]
        model_path = model_info["path"]
        
        print(f"\n{'='*60}")
        print(f"Loading Model: {model_info['name']}")
        print(f"Path: {model_path}")
        print(f"{'='*60}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            print("✓ Model checkpoint loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading model checkpoint: {e}")
        
        # Create model
        self.model = StreamlinedSequenceGenerator(
            input_size=7,
            hidden_size=256,
            num_layers=2,
            output_seq_len=151,
            dropout=0.3
        ).to(self.device)
        
        # Load weights
        try:
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Print training info if available
                if 'epoch' in checkpoint:
                    print(f"✓ Model trained for {checkpoint['epoch'] + 1} epochs")
                if 'val_loss' in checkpoint:
                    print(f"✓ Best validation loss: {checkpoint['val_loss']:.6f}")
            else:
                self.model.load_state_dict(checkpoint)
            
            print("✓ Model weights loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {e}")
        
        self.model.eval()
        
    def predict_sequence(self, initial_values):
        """
        Generate a full sequence from initial component values.
        
        Args:
            initial_values (np.ndarray): Initial component values (7 values)
            
        Returns:
            np.ndarray: Predicted sequence (151, 7)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        print(f"\n{'='*60}")
        print("STEP 1: Generating Sequence Prediction")
        print(f"{'='*60}")
        print(f"Initial values: {initial_values}")
        
        # Prepare input tensor
        input_tensor = torch.tensor(initial_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Generate prediction
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Convert to numpy
        predicted_sequence = prediction.squeeze(0).cpu().numpy()
        
        # Force the first step to match the input values
        predicted_sequence[0] = initial_values
        
        self.prediction = predicted_sequence
        
        print(f"✓ Sequence generated: {predicted_sequence.shape[0]} time steps")
        
        return predicted_sequence
    
    def save_prediction(self, initial_values):
        """
        Save prediction to CSV and generate plots.
        
        Args:
            initial_values (np.ndarray): Initial component values for naming
        """
        if self.prediction is None:
            raise RuntimeError("No prediction available. Call predict_sequence() first.")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        initial_str = "_".join([f"{v:.4f}" for v in initial_values[:3]])  # Use first 3 components for naming
        self.output_dir = os.path.join(self.output_base_dir, f"{timestamp}_{initial_str}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("Saving Prediction Results")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        
        # Save to CSV
        time_steps = np.arange(151)
        df = pd.DataFrame(self.prediction, columns=self.COMPONENT_NAMES)
        df.insert(0, 'step', time_steps)
        
        prediction_file = os.path.join(self.output_dir, "prediction.csv")
        df.to_csv(prediction_file, index=False)
        print(f"✓ Prediction saved to: prediction.csv")
        
        # Save initial values for reference
        initial_df = pd.DataFrame([initial_values], columns=self.COMPONENT_NAMES)
        initial_file = os.path.join(self.output_dir, "initial_values.csv")
        initial_df.to_csv(initial_file, index=False)
        print(f"✓ Initial values saved to: initial_values.csv")
        
        # Generate plots (skip mobility and gradient_coefficient as they are constants)
        print("\nGenerating plots...")
        plot_count = 0
        for i, name in enumerate(self.COMPONENT_NAMES):
            # Skip mobility and gradient_coefficient
            if name in ['mobility', 'gradient_coefficient']:
                continue
                
            plt.figure(figsize=(12, 6))
            plt.plot(time_steps, self.prediction[:, i], 'b-', linewidth=2, label='Predicted')
            plt.scatter([0], [initial_values[i]], color='red', s=100, zorder=5, label='Initial')
            plt.title(f'{name} - Sequence Prediction', fontsize=14, fontweight='bold')
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_file = os.path.join(self.output_dir, f"{name}_prediction.png")
            plt.savefig(plot_file, dpi=150)
            plt.close()
            plot_count += 1
        
        print(f"✓ {plot_count} component plots saved (mobility and gradient_coefficient skipped as constants)")
        
        return prediction_file
    
    def deduce_time(self, target_values, temperature=0.1, top_k=10):
        """
        Deduce the most likely time step for target component values.
        
        Args:
            target_values (np.ndarray): Target component values (7 values, can contain NaN)
            temperature (float): Softmax temperature (lower = more peaked distribution)
            top_k (int): Number of top results to return
            
        Returns:
            tuple: (top_indices, top_probabilities)
        """
        if self.prediction is None:
            raise RuntimeError("No prediction available. Call predict_sequence() first.")
        
        print(f"\n{'='*60}")
        print("STEP 2: Deducing Time Position")
        print(f"{'='*60}")
        print(f"Target values: {target_values}")
        
        # Handle missing values (NaN)
        valid_mask = ~np.isnan(target_values)
        print(f"Valid components: {np.sum(valid_mask)} out of {len(target_values)}")
        
        # Calculate distances for each time step
        distances = np.zeros(len(self.prediction))
        for step in range(len(self.prediction)):
            step_values = self.prediction[step]
            
            # Calculate weighted squared difference for valid components
            squared_diff = 0
            for i in range(len(target_values)):
                if valid_mask[i]:
                    component_diff = (step_values[i] - target_values[i]) ** 2
                    squared_diff += component_diff * self.COMPONENT_WEIGHTS[i]
            
            distances[step] = squared_diff
        
        # Convert distances to probabilities using softmax
        negative_distances = -distances / temperature
        probabilities = softmax(negative_distances)
        
        # Get top K most likely time steps
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_probabilities = probabilities[top_indices]
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Top {top_k} Most Likely Time Steps")
        print(f"{'='*60}")
        
        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probabilities)):
            step_values = self.prediction[idx]
            
            print(f"\nRank {i+1}: Step {idx}, Probability: {prob:.6f} ({prob*100:.2f}%)")
            
            result = {
                "Rank": i+1,
                "Step": int(idx),
                "Probability": prob,
                "Probability_%": prob * 100
            }
            
            # Print component-wise comparison
            for j, comp in enumerate(self.COMPONENT_NAMES):
                if valid_mask[j]:
                    diff = abs(target_values[j] - step_values[j])
                    print(f"  {comp:20s}: Target={target_values[j]:8.4f}, Predicted={step_values[j]:8.4f}, Diff={diff:8.4f}")
                    
                    result[f"{comp}_Target"] = target_values[j]
                    result[f"{comp}_Predicted"] = step_values[j]
                    result[f"{comp}_Diff"] = diff
                else:
                    print(f"  {comp:20s}: Target=N/A, Predicted={step_values[j]:8.4f}")
                    
                    result[f"{comp}_Target"] = np.nan
                    result[f"{comp}_Predicted"] = step_values[j]
                    result[f"{comp}_Diff"] = np.nan
            
            results.append(result)
        
        # Save results
        self._save_deduction_results(results, top_indices, top_probabilities, target_values)
        
        return top_indices, top_probabilities
    
    def _save_deduction_results(self, results, top_indices, top_probabilities, target_values):
        """Save time deduction results to CSV and generate plots"""
        if self.output_dir is None:
            raise RuntimeError("Output directory not set. Call save_prediction() first.")
        
        print(f"\n{'='*60}")
        print("Saving Time Deduction Results")
        print(f"{'='*60}")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_file = os.path.join(self.output_dir, "time_deduction_results.csv")
        results_df.to_csv(results_file, index=False)
        print(f"✓ Results saved to: time_deduction_results.csv")
        
        # Save target values for reference
        valid_mask = ~np.isnan(target_values)
        target_df = pd.DataFrame([target_values], columns=self.COMPONENT_NAMES)
        target_file = os.path.join(self.output_dir, "target_values.csv")
        target_df.to_csv(target_file, index=False)
        print(f"✓ Target values saved to: target_values.csv")
        
        # Plot probability distribution
        plt.figure(figsize=(14, 6))
        plt.bar(top_indices, top_probabilities, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('Probability Distribution of Most Likely Time Steps', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        prob_plot_file = os.path.join(self.output_dir, "probability_distribution.png")
        plt.savefig(prob_plot_file, dpi=150)
        plt.close()
        print(f"✓ Probability plot saved")
        
        # Plot component-wise comparison for top result
        top_idx = top_indices[0]
        top_step_values = self.prediction[top_idx]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, comp in enumerate(self.COMPONENT_NAMES):
            ax = axes[i]
            
            if valid_mask[i]:
                # Plot target vs predicted
                ax.bar(['Target', 'Predicted'], [target_values[i], top_step_values[i]], 
                       color=['orange', 'skyblue'], edgecolor='black', alpha=0.7)
                ax.set_ylabel('Value', fontsize=10)
                ax.set_title(f'{comp}\n(Diff: {abs(target_values[i] - top_step_values[i]):.4f})', 
                            fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            else:
                # Only show predicted value
                ax.bar(['Predicted'], [top_step_values[i]], 
                       color=['skyblue'], edgecolor='black', alpha=0.7)
                ax.set_ylabel('Value', fontsize=10)
                ax.set_title(f'{comp}\n(Target: N/A)', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
        
        # Remove extra subplot
        fig.delaxes(axes[7])
        
        plt.suptitle(f'Component Comparison - Top Result (Step {top_idx})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        comp_plot_file = os.path.join(self.output_dir, "component_comparison.png")
        plt.savefig(comp_plot_file, dpi=150)
        plt.close()
        print(f"✓ Component comparison plot saved")
        
        # Generate comprehensive all-in-one plot with target time highlighted
        self._plot_all_components_with_target(top_idx, target_values, valid_mask)
    
    def _plot_all_components_with_target(self, target_step, target_values, valid_mask):
        """
        Create a comprehensive plot showing all components with the target time highlighted.
        Only plots the 5 PCA components (excludes mobility and gradient_coefficient as they are constants).
        
        Args:
            target_step (int): The predicted target time step
            target_values (np.ndarray): Target component values
            valid_mask (np.ndarray): Boolean mask for valid target components
        """
        time_steps = np.arange(151)
        
        # Only plot the 5 PCA components (skip mobility and gradient_coefficient)
        pca_components = ['comp_1', 'comp_2', 'comp_3', 'comp_4', 'comp_5']
        pca_indices = [i for i, name in enumerate(self.COMPONENT_NAMES) if name in pca_components]
        
        # Create figure with subplots for 5 components
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for plot_idx, comp_idx in enumerate(pca_indices):
            comp = self.COMPONENT_NAMES[comp_idx]
            ax = axes[plot_idx]
            
            # Plot the full predicted sequence
            ax.plot(time_steps, self.prediction[:, comp_idx], 'b-', linewidth=2, label='Predicted Sequence', alpha=0.7)
            
            # Highlight the target time step
            ax.axvline(x=target_step, color='red', linestyle='--', linewidth=2, label=f'Target Time (Step {target_step})', alpha=0.8)
            
            # Mark the predicted value at target time
            predicted_at_target = self.prediction[target_step, comp_idx]
            ax.scatter([target_step], [predicted_at_target], color='red', s=150, zorder=5, 
                      marker='o', edgecolors='darkred', linewidths=2, label=f'Predicted @ Step {target_step}')
            
            # If target value is available, show it
            if valid_mask[comp_idx]:
                ax.scatter([target_step], [target_values[comp_idx]], color='orange', s=150, zorder=6,
                          marker='*', edgecolors='darkorange', linewidths=2, label='Target Value')
                
                # Add difference annotation
                diff = abs(target_values[comp_idx] - predicted_at_target)
                ax.text(target_step, target_values[comp_idx], f'  Δ={diff:.4f}', 
                       fontsize=9, verticalalignment='bottom', color='darkred', fontweight='bold')
            
            # Mark initial point
            ax.scatter([0], [self.prediction[0, comp_idx]], color='green', s=100, zorder=5,
                      marker='s', edgecolors='darkgreen', linewidths=2, label='Initial Value')
            
            ax.set_xlabel('Time Step', fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=11, fontweight='bold')
            ax.set_title(f'{comp}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            
            # Add shaded region around target time
            ax.axvspan(max(0, target_step-5), min(150, target_step+5), alpha=0.1, color='red')
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.suptitle(f'PCA Components Prediction with Target Time Highlighted\nPredicted Target Time: Step {target_step}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        all_comp_plot_file = os.path.join(self.output_dir, "all_components_with_target.png")
        plt.savefig(all_comp_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ All PCA components plot with target time saved")
        
        # Also create a single-page overview with all components overlaid
        self._plot_all_components_overlaid(target_step, target_values, valid_mask)
    
    def _plot_all_components_overlaid(self, target_step, target_values, valid_mask):
        """
        Create a single plot with all components overlaid (normalized).
        PCA components are plotted as curves, mobility and gradient_coefficient as constant horizontal lines.
        
        Args:
            target_step (int): The predicted target time step
            target_values (np.ndarray): Target component values
            valid_mask (np.ndarray): Boolean mask for valid target components
        """
        time_steps = np.arange(151)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        
        # Get initial values for constants
        mobility_value = self.prediction[0, 5]  # mobility is index 5
        gradient_coef_value = self.prediction[0, 6]  # gradient_coefficient is index 6
        
        # Normalize each PCA component to [0, 1] for visualization
        for i, comp in enumerate(self.COMPONENT_NAMES):
            # Skip mobility and gradient_coefficient - they will be plotted as constants
            if comp in ['mobility', 'gradient_coefficient']:
                continue
                
            component_data = self.prediction[:, i]
            
            # Normalize
            min_val = component_data.min()
            max_val = component_data.max()
            if max_val - min_val > 1e-6:  # Avoid division by zero
                normalized = (component_data - min_val) / (max_val - min_val)
            else:
                normalized = component_data
            
            ax.plot(time_steps, normalized, color=colors[i], linewidth=2, label=comp, alpha=0.8)
        
        # Plot mobility and gradient_coefficient as constant horizontal lines
        # Normalize them to fit in [0, 1] range for visualization
        all_values = []
        for i in range(5):  # Only PCA components
            all_values.extend([self.prediction[:, i].min(), self.prediction[:, i].max()])
        
        overall_min = min(all_values)
        overall_max = max(all_values)
        
        if overall_max - overall_min > 1e-6:
            mobility_normalized = (mobility_value - overall_min) / (overall_max - overall_min)
            gradient_normalized = (gradient_coef_value - overall_min) / (overall_max - overall_min)
        else:
            mobility_normalized = 0.5
            gradient_normalized = 0.5
        
        ax.axhline(y=mobility_normalized, color=colors[5], linestyle='-', linewidth=2.5, 
                  label=f'mobility (constant = {mobility_value:.4f})', alpha=0.8)
        ax.axhline(y=gradient_normalized, color=colors[6], linestyle='-', linewidth=2.5, 
                  label=f'gradient_coefficient (constant = {gradient_coef_value:.4f})', alpha=0.8)
        
        # Highlight target time
        ax.axvline(x=target_step, color='red', linestyle='--', linewidth=3, 
                  label=f'Target Time (Step {target_step})', alpha=0.9, zorder=10)
        ax.axvspan(max(0, target_step-5), min(150, target_step+5), alpha=0.15, color='red', zorder=1)
        
        ax.set_xlabel('Time Step', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Value [0-1]', fontsize=14, fontweight='bold')
        ax.set_title(f'All Components Overlaid (Normalized) - Target Time: Step {target_step}', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10, ncol=2)
        
        plt.tight_layout()
        
        overlay_plot_file = os.path.join(self.output_dir, "all_components_overlay.png")
        plt.savefig(overlay_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ All components overlay plot saved (PCA curves + constant lines for mobility & gradient_coefficient)")


def parse_component_input(input_str):
    """
    Parse component input string to numpy array.
    Supports comma-separated values, with 'nan' or 'N/A' for missing values.
    
    Args:
        input_str (str): Input string (e.g., "1.2,3.4,5.6,nan,7.8,9.0,nan")
        
    Returns:
        np.ndarray: Array of 7 float values (with NaN for missing)
    """
    values = []
    parts = input_str.split(',')
    
    if len(parts) != 7:
        raise ValueError(f"Expected 7 values, got {len(parts)}")
    
    for part in parts:
        part = part.strip().lower()
        if part in ['nan', 'n/a', 'na', '']:
            values.append(np.nan)
        else:
            try:
                values.append(float(part))
            except ValueError:
                raise ValueError(f"Invalid value: {part}")
    
    return np.array(values)


def interactive_mode():
    """Run the pipeline in interactive mode"""
    print("\n" + "="*60)
    print(" SEQUENCE PREDICTION & TIME DEDUCTION PIPELINE")
    print("="*60)
    
    # Step 1: Select model
    print("\nAvailable Models:")
    for key, info in PredictionAndDeductionPipeline.MODELS.items():
        print(f"  {key}. {info['name']}")
    
    while True:
        model_choice = input("\nSelect model (1 or 2): ").strip()
        if model_choice in PredictionAndDeductionPipeline.MODELS:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    # Step 2: Get initial component values
    print("\n" + "-"*60)
    print("Enter initial component values (7 values, comma-separated)")
    print("Format: comp_1,comp_2,comp_3,comp_4,comp_5,mobility,gradient_coefficient")
    print("Example: 1.343466,3.098296,-2.72734,-1.76372,-1.44867,0.379756,1.227265")
    
    while True:
        try:
            initial_input = input("\nInitial values: ").strip()
            initial_values = parse_component_input(initial_input)
            if np.any(np.isnan(initial_values)):
                print("Error: Initial values cannot contain NaN. Please provide all 7 values.")
                continue
            break
        except ValueError as e:
            print(f"Error: {e}. Please try again.")
    
    # Step 3: Get target component values
    print("\n" + "-"*60)
    print("Enter target component values (7 values, comma-separated)")
    print("Use 'nan' or 'N/A' for unknown components")
    print("Example: 0.281879,-2.27557,-0.53413,-0.08212,0.162859,nan,nan")
    
    while True:
        try:
            target_input = input("\nTarget values: ").strip()
            target_values = parse_component_input(target_input)
            break
        except ValueError as e:
            print(f"Error: {e}. Please try again.")
    
    # Step 4: Optional parameters
    print("\n" + "-"*60)
    output_dir = input("Output directory (press Enter for default 'D:/results/predictions'): ").strip()
    if not output_dir:
        output_dir = "D:/results/predictions"
    
    # Run pipeline
    print("\n" + "="*60)
    print(" RUNNING PIPELINE")
    print("="*60)
    
    try:
        pipeline = PredictionAndDeductionPipeline(model_choice, output_dir)
        pipeline.load_model()
        pipeline.predict_sequence(initial_values)
        prediction_file = pipeline.save_prediction(initial_values)
        top_indices, top_probs = pipeline.deduce_time(target_values)
        
        # Print final summary
        print("\n" + "="*60)
        print(" PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n🎯 PREDICTED TARGET TIME: Step {top_indices[0]}")
        print(f"   Confidence: {top_probs[0]*100:.2f}%")
        print(f"\n📊 Top 3 Most Likely Time Steps:")
        for i in range(min(3, len(top_indices))):
            print(f"   {i+1}. Step {top_indices[i]:3d} - Probability: {top_probs[i]*100:6.2f}%")
        print(f"\n📁 All results saved to:")
        print(f"   {pipeline.output_dir}")
        print(f"\n📈 Generated plots:")
        print(f"   • all_components_with_target.png - 5 PCA component curves with target highlighted")
        print(f"   • all_components_overlay.png - PCA curves + constant lines (mobility & gradient_coefficient)")
        print(f"   • probability_distribution.png - Probability distribution")
        print(f"   • component_comparison.png - Target vs predicted comparison")
        print(f"   • comp_1 to comp_5 plots - Individual PCA component predictions")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def batch_mode(model_choice, initial_values, target_values, output_dir="D:/results/predictions"):
    """
    Run the pipeline in batch mode (for scripting).
    
    Args:
        model_choice (str): Model selection ("1" or "2")
        initial_values (np.ndarray or list): Initial component values (7 values)
        target_values (np.ndarray or list): Target component values (7 values, can contain NaN)
        output_dir (str): Output directory
        
    Returns:
        tuple: (top_indices, top_probabilities, output_directory)
    """
    initial_values = np.array(initial_values)
    target_values = np.array(target_values)
    
    pipeline = PredictionAndDeductionPipeline(model_choice, output_dir)
    pipeline.load_model()
    pipeline.predict_sequence(initial_values)
    pipeline.save_prediction(initial_values)
    top_indices, top_probabilities = pipeline.deduce_time(target_values)
    
    return top_indices, top_probabilities, pipeline.output_dir


if __name__ == "__main__":
    # Check if running in batch mode (with command-line arguments)
    if len(sys.argv) > 1:
        print("Batch mode not implemented via command line.")
        print("Use interactive_mode() or batch_mode() function directly.")
        sys.exit(1)
    
    # Run in interactive mode
    interactive_mode()
