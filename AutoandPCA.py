import os
import numpy as np
from sklearn.decomposition import PCA
import vtk
from scipy.spatial.distance import pdist, squareform
from scipy.signal import correlate2d
import matplotlib.pyplot as plt
from vtk.util import numpy_support


def get_parameters_from_folder(folder_name):
    """Extract parameters from folder name"""
    try:
        # Split folder name by comma
        params = folder_name.split(',')
        if len(params) == 3:
            return tuple(float(p) for p in params)
    except:
        pass
    return None


def create_output_folder(input_folder, batch_name, base_output_path="D:/sample1"):
    """Create corresponding output folder with batch organization"""
    folder_name = os.path.basename(input_folder)
    batch_dir = os.path.join(base_output_path, batch_name)
    os.makedirs(batch_dir, exist_ok=True)
    output_folder = os.path.join(batch_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def analyze_vtk_file(vtk_file):
    """Analyze a VTK file structure and content"""
    try:
        with open(vtk_file, 'r') as f:
            lines = f.readlines()
            
        analysis = {
            'total_lines': len(lines),
            'header': [],
            'dimensions': None,
            'data_section': None,
            'data_count': 0,
            'non_zero_count': 0,
            'has_nan': False,
            'file_size': os.path.getsize(vtk_file)
        }
        
        # Analyze header and structure
        for i, line in enumerate(lines[:20]):  # Look at first 20 lines
            analysis['header'].append(line.strip())
            if 'DIMENSIONS' in line:
                try:
                    dims = [int(x) for x in line.split()[1:]]
                    analysis['dimensions'] = dims
                    analysis['expected_points'] = np.prod(dims)
                except:
                    pass
            elif 'POINT_DATA' in line:
                analysis['data_section'] = i + 1
                
        # Analyze data content
        if analysis['data_section'] is not None:
            data_values = []
            for line in lines[analysis['data_section']:]:
                try:
                    val = float(line.strip())
                    data_values.append(val)
                    if val != 0:
                        analysis['non_zero_count'] += 1
                    if np.isnan(val):
                        analysis['has_nan'] = True
                except ValueError:
                    continue
            
            analysis['data_count'] = len(data_values)
            if data_values:
                analysis['min_value'] = np.min(data_values)
                analysis['max_value'] = np.max(data_values)
                analysis['mean_value'] = np.mean(data_values)
                analysis['std_value'] = np.std(data_values)
            
        return analysis
    except Exception as e:
        return {'error': str(e)}


def load_vtk_files(folder_path):
    """Load VTK files from a folder with detailed analysis"""
    vtk_files = []
    timesteps = []
    
    print(f"\nAnalyzing VTK files in: {folder_path}")
    
    # Find all VTK files
    for file in os.listdir(folder_path):
        if file.endswith('.vtk') and file.startswith('time_'):
            vtk_files.append(os.path.join(folder_path, file))
            try:
                # Extract timestep as integer for proper numerical sorting
                timestep = int(file.split('_')[1].split('.')[0])
                timesteps.append(timestep)
            except ValueError:
                print(f"Error parsing timestep from file: {file}")
                continue
    
    if not vtk_files:
        print("No VTK files found!")
        return [], []
    
    # Sort files by timestep numerically
    sorted_indices = np.argsort(timesteps)
    sorted_vtk_files = [vtk_files[i] for i in sorted_indices]
    sorted_timesteps = [timesteps[i] for i in sorted_indices]
    
    print(f"Found {len(sorted_vtk_files)} VTK files")
    print("Files will be processed in this timestep order:")
    for ts in sorted_timesteps:
        print(f"  - Timestep {ts}")
    
    data_list = []
    valid_timesteps = []
    
    for vtk_file, timestep in zip(sorted_vtk_files, sorted_timesteps):
        try:
            # Read the file content
            with open(vtk_file, 'r') as f:
                lines = f.readlines()
            
            dimensions = None
            data_values = []
            reading_data = False
            data_type = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                
                if 'DIMENSIONS' in line:
                    try:
                        dims = line.split()[1:]
                        dimensions = [int(d) for d in dims]
                    except:
                        continue
                        
                elif 'SCALARS' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        data_type = parts[2]
                        reading_data = False  # Wait for LOOKUP_TABLE
                        
                elif 'LOOKUP_TABLE' in line:
                    reading_data = True
                    continue
                    
                elif reading_data and line:
                    try:
                        vals = [float(v) for v in line.split()]
                        data_values.extend(vals)
                    except ValueError:
                        continue
            
            if not dimensions:
                print(f"\nNo dimensions found in {os.path.basename(vtk_file)}")
                continue
                
            if not data_values:
                print(f"\nNo valid data values found in {os.path.basename(vtk_file)}")
                continue
            
            # Convert to numpy array and validate
            data = np.array(data_values, dtype=np.float64)
            expected_size = np.prod(dimensions)
            
            if data.size != expected_size:
                print(f"\nData size mismatch in {os.path.basename(vtk_file)}")
                print(f"Expected {expected_size} points, got {data.size}")
                continue
            
            # Reshape and validate
            try:
                data = data.reshape(dimensions)
                
                # Print some statistics about the data
                print(f"\nFile: {os.path.basename(vtk_file)}")
                print(f"- Shape: {data.shape}")
                print(f"- Non-zero values: {np.count_nonzero(data)}")
                print(f"- Value range: {np.min(data)} to {np.max(data)}")
                
                data_list.append(data)
                valid_timesteps.append(timestep)
                
            except Exception as e:
                print(f"\nError reshaping data in {os.path.basename(vtk_file)}: {str(e)}")
                continue
                
        except Exception as e:
            print(f"\nError processing {os.path.basename(vtk_file)}: {str(e)}")
            continue
    
    if not data_list:
        print("\nNo valid data could be loaded from any VTK file!")
        print("Please check the file format and content.")
    else:
        print(f"\nSuccessfully loaded {len(data_list)} out of {len(sorted_vtk_files)} files")
    
    return data_list, valid_timesteps


def calculate_autocorrelation(phase_field):
    """Calculate 2D spatial autocorrelation using FFT method"""
    try:
        # Ensure input is 2D
        if len(phase_field.shape) == 3:
            phase_field = np.mean(phase_field, axis=2)
        
        # Normalize the data
        phase_field = phase_field - np.mean(phase_field)
        std_dev = np.std(phase_field)
        if std_dev > 1e-10:
            phase_field = phase_field / std_dev
            
        # Calculate autocorrelation using FFT
        fft_data = np.fft.fft2(phase_field)
        autocorr = np.real(np.fft.ifft2(fft_data * np.conj(fft_data)))
        
        # Normalize and shift
        autocorr = np.fft.fftshift(autocorr)
        autocorr = autocorr / autocorr.max()
        
        # Calculate radial average
        center = (autocorr.shape[0] // 2, autocorr.shape[1] // 2)
        y, x = np.indices(autocorr.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # Calculate radial profile
        tbin = np.bincount(r.ravel(), autocorr.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / nr
        
        return autocorr, radial_profile
        
    except Exception as e:
        print(f"Error in autocorrelation calculation: {str(e)}")
        return None, None


def process_timestep(data, timestep):
    """Process a single timestep of data"""
    print(f"\nProcessing timestep {timestep}")
    
    # Calculate autocorrelation and radial profile
    autocorr, radial_profile = calculate_autocorrelation(data)
    if autocorr is None:
        return None
    
    # Flatten autocorrelation for PCA input
    flattened_autocorr = autocorr.flatten()
    
    return {
        'timestep': timestep,
        'autocorrelation': autocorr,
        'radial_profile': radial_profile,
        'flattened_autocorr': flattened_autocorr
    }


def process_folder(input_folder, output_folder):
    """Process a single parameter folder"""
    try:
        folder_name = os.path.basename(input_folder)
        print(f"\nProcessing folder: {folder_name}")
        
        # Load VTK files
        data_arrays, timesteps = load_vtk_files(input_folder)
        
        if not data_arrays or not timesteps:
            print(f"No valid data found in folder: {folder_name}")
            return
        
        print(f"Processing {len(data_arrays)} timesteps...")
        
        # Process each timestep
        results = []
        flattened_autocorrs = []
        original_timesteps = []  # Store original timesteps to maintain order
        
        for i, (data, timestep) in enumerate(zip(data_arrays, timesteps)):
            print(f"Processing timestep {timestep} ({i+1}/{len(data_arrays)})")
            
            result = process_timestep(data, timestep)
            if result is not None:
                results.append(result)
                flattened_autocorrs.append(result['flattened_autocorr'])
                original_timesteps.append(timestep)  # Keep track of the timestep
        
        # Perform PCA on all autocorrelations
        if flattened_autocorrs:
            X = np.array(flattened_autocorrs)
            original_timesteps = np.array(original_timesteps)  # Convert to numpy array for easier indexing
            
            # Check if timesteps are already sorted (they should be from load_vtk_files)
            if not np.all(np.diff(original_timesteps) >= 0):
                print("WARNING: Timesteps are not in ascending order. Sorting now...")
                # Sort timesteps and corresponding data to ensure consistent ordering
                sort_indices = np.argsort(original_timesteps)
                X = X[sort_indices]
                original_timesteps = original_timesteps[sort_indices]
                # Also sort results to maintain consistency
                results = [results[i] for i in sort_indices]
            
            # PCA on correctly ordered data
            pca_full = PCA()
            pca_full.fit(X)
            
            # Find number of components needed for 99% variance
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            
            # Fixed component count of 5 instead of dynamic calculation
            n_components = 5
            
            # Show original calculation for reference
            original_components = np.argmax(cumsum >= 0.99) + 1
            
            print(f"\nAnalysis of variance explained:")
            print(f"First component explains: {pca_full.explained_variance_ratio_[0]:.4%}")
            print(f"Originally needed {original_components} components for 99% variance")
            print(f"Using fixed component count: {n_components}")
            
            # Now perform PCA with exactly the number of components needed
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(X)
            
            print("\nFinal PCA Results:")
            print(f"Number of components used: {n_components}")
            print(f"Individual variance ratios:")
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                print(f"  Component {i+1}: {ratio:.4%}")
            print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.4%}")
            
            # Print timestep order for verification
            print("\nTimestep order in PCA components:")
            for i, ts in enumerate(original_timesteps):
                print(f"  Row {i} corresponds to timestep {ts}")
            
            # Save results with timestep information
            save_results(output_folder, results, pca_result, pca.explained_variance_ratio_, original_timesteps)
            print(f"Results saved in: {output_folder}")
        else:
            print("No valid results to save")
            
    except Exception as e:
        print(f"\nError processing folder {folder_name}: {str(e)}")


def save_results(output_folder, results, pca_result, variance_ratios, timesteps):
    """Save analysis results"""
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        # Save PCA results with timestep mapping
        np.savez(os.path.join(output_folder, 'pca_results.npz'),
                 components=pca_result,
                 variance_ratios=variance_ratios,
                 timesteps=timesteps)  # Add timesteps to understand the order
        
        # Create a timestep index map file for reference
        with open(os.path.join(output_folder, 'timestep_index_map.txt'), 'w') as f:
            f.write('Timestep to Index Mapping\n')
            f.write('=======================\n\n')
            for i, ts in enumerate(timesteps):
                f.write(f'Timestep {ts} -> Index {i}\n')
        
        # Save autocorrelations and radial profiles
        for result in results:
            timestep = result['timestep']
            np.savez(os.path.join(output_folder, f'timestep_{timestep}.npz'),
                    autocorrelation=result['autocorrelation'],
                    radial_profile=result['radial_profile'])
        
        # Save summary
        with open(os.path.join(output_folder, 'summary.txt'), 'w') as f:
            f.write('Analysis Summary\n')
            f.write('===============\n\n')
            f.write(f'Total timesteps: {len(results)}\n')
            f.write(f'PCA components: {len(variance_ratios)}\n')
            f.write('Explained variance ratios:\n')
            for i, ratio in enumerate(variance_ratios):
                f.write(f'Component {i+1}: {ratio:.4%}\n')
            f.write(f'\nTotal variance explained: {np.sum(variance_ratios):.4%}\n')
            f.write('\nTimestep ordering in PCA components:\n')
            for i, ts in enumerate(timesteps):
                f.write(f'Row {i} corresponds to timestep {ts}\n')
            
    except Exception as e:
        print(f"Error saving results: {str(e)}")


def main(batch_name):
    """Process all simulations in a specified batch"""
    try:
        input_base = os.path.join("D:/sample", batch_name)
        
        # Validate input directory exists
        if not os.path.exists(input_base):
            print(f"Error: Input directory {input_base} does not exist")
            return
        
        # List all parameter folders in the batch
        param_folders = [d for d in os.listdir(input_base) 
                        if os.path.isdir(os.path.join(input_base, d))]
        
        if not param_folders:
            print(f"No parameter folders found in {input_base}")
            return
        
        print(f"Found {len(param_folders)} parameter folders to process in batch {batch_name}")
        
        # Process each parameter folder
        for i, folder in enumerate(param_folders, 1):
            input_folder = os.path.join(input_base, folder)
            print(f"\nProcessing folder {i}/{len(param_folders)}")
            
            # Create output folder in the corresponding batch directory
            output_folder = create_output_folder(input_folder, batch_name)
            process_folder(input_folder, output_folder)
        
        print(f"\nAnalysis completed for batch {batch_name}")
        print(f"Results saved in: D:/sample1/{batch_name}")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")


def list_available_batches(base_dir="D:/sample"):
    """List all available batch folders in D:/sample"""
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist")
        return []
    
    batches = [d for d in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, d))]
    return sorted(batches)


if __name__ == "__main__":
    # List available batches
    available_batches = list_available_batches()
    
    if not available_batches:
        print("No batch folders found in D:/sample")
        exit()
    
    print("\nAvailable batches in D:/sample:")
    for batch in available_batches:
        print(f"- {batch}")
    
    # Ask for batch name
    while True:
        batch_name = input("\nEnter batch name to process (e.g., '1'): ").strip()
        if batch_name in available_batches:
            break
        print(f"Error: Batch '{batch_name}' not found in D:/sample")
    
    # Process the selected batch
    main(batch_name)
