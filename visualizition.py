import vtk
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from vtk.util import numpy_support
import glob


# Function to read VTK file and extract concentration data
def read_vtk_file(vtk_filename):
    # Create a generic reader
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(vtk_filename)
    reader.Update()
    
    # Get the output data
    output = reader.GetOutput()
    
    # Get point data
    point_data = output.GetPointData()
    
    # Get the concentration array (named 'CON')
    con_array = point_data.GetArray('CON')
    
    # Convert to numpy array
    numpy_array = numpy_support.vtk_to_numpy(con_array)
    
    # Get dimensions
    dimensions = output.GetDimensions()
    nx, ny, nz = dimensions
    
    # Reshape to 2D array
    if nz == 1:  # 2D data
        concentration_2d = numpy_array.reshape((ny, nx))
    else:  # 3D data, take middle slice
        concentration_2d = numpy_array.reshape((nz, ny, nx))[nz//2, :, :]
    
    return concentration_2d


# Function to create animation from VTK files
def create_vtk_animation(folder_path, interval=200, step_interval=1000):
    """
    Create a simple animation of concentration data from VTK files.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing VTK files
    interval : int, optional
        Interval between frames in milliseconds
    step_interval : int, optional
        Interval between steps to include in the animation
    """
    # Get list of VTK files sorted by timestep
    vtk_files = sorted(glob.glob(os.path.join(folder_path, "*.vtk")), 
                      key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    if not vtk_files:
        print(f"No VTK files found in {folder_path}")
        return
    
    # Filter files based on step interval
    filtered_files = vtk_files[::step_interval]
    if len(filtered_files) > 30:  # Limit to 30 frames for performance
        filtered_files = filtered_files[:30]
    
    print(f"Using {len(filtered_files)} files for animation")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Read first file to initialize plot
    first_data = read_vtk_file(filtered_files[0])
    
    # Create initial plot
    img = ax.imshow(first_data, cmap='plasma', origin='lower')
    fig.colorbar(img, ax=ax, label='Concentration')
    
    # Get the first timestep
    first_timestep = int(os.path.basename(filtered_files[0]).split('_')[1].split('.')[0])
    title = ax.set_title(f"Time Step: {first_timestep}", fontsize=14)
    
    # Update function for animation
    def update(frame):
        # Read the VTK file for this frame
        data = read_vtk_file(filtered_files[frame])
        
        # Update the image data
        img.set_array(data)
        
        # Get the current timestep
        timestep = int(os.path.basename(filtered_files[frame]).split('_')[1].split('.')[0])
        
        # Update the title with the current timestep
        title.set_text(f"Time Step: {timestep}")
        
        return img, title
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(filtered_files), interval=interval, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim


# Example usage
if __name__ == "__main__":
    # Path to the folder containing VTK files
    folder_path = r'D:\sample\56\0.658266891531958,1.0996563939513853,0.8654710422772286'
    
    # Create and display animation
    animation = create_vtk_animation(
        folder_path=folder_path,
        interval=200,  # Milliseconds between frames
        step_interval=5  # Include every 5th file
    )
