import os
import shutil

def process_pca_files(source_dir, dest_dir):
    # Walk through the source directory
    for main_folder in os.listdir(source_dir):
        main_path = os.path.join(source_dir, main_folder)
        
        # Check if it's a directory
        if not os.path.isdir(main_path):
            continue
        
        # Create corresponding main folder in destination
        dest_main_path = os.path.join(dest_dir, main_folder)
        os.makedirs(dest_main_path, exist_ok=True)
        
        # Walk through subfolders
        for subfolder in os.listdir(main_path):
            subfolder_path = os.path.join(main_path, subfolder)
            
            # Check if it's a directory
            if not os.path.isdir(subfolder_path):
                continue
            
            # Split subfolder name and get 2nd and 3rd numbers
            try:
                # Split by comma and get the 2nd and 3rd numbers
                parts = subfolder.split(',')
                if len(parts) < 3:
                    print(f"Skipping invalid subfolder: {subfolder}")
                    continue
                
                # Create new filename with 2nd and 3rd numbers
                new_filename = f"{parts[1]},{parts[2]}.npz"
                
                # Full path to pca_results.npz
                pca_file = os.path.join(subfolder_path, 'pca_results.npz')
                
                # Check if pca_results.npz exists
                if not os.path.exists(pca_file):
                    print(f"No pca_results.npz found in {subfolder_path}")
                    continue
                
                # Create destination path
                dest_subfolder_path = os.path.join(dest_main_path, new_filename)
                
                # Copy the file
                shutil.copy2(pca_file, dest_subfolder_path)
                print(f"Copied {pca_file} to {dest_subfolder_path}")
            
            except Exception as e:
                print(f"Error processing {subfolder}: {e}")

def main():
    source_dir = r'D:\sample1'
    dest_dir = r'D:\sample2'
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Process PCA files
    process_pca_files(source_dir, dest_dir)
    print("File processing complete!")

if __name__ == '__main__':
    main()
