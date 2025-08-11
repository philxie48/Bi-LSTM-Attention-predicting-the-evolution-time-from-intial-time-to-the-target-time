import numpy as np
import os
import random


# Function to initialize the concentration field with random fluctuations
def micro_ch_pre(Nx, Ny, c0, iflag, noise=0.1):
    """
    Initializes the concentration field using stronger random noise around the mean c0.
    """
    con = c0 + noise * (np.random.rand(Nx, Ny) - 0.5)  # Increased noise amplitude
    return con if iflag == 1 else con.flatten()


def dfdc(c):
    """
    Free energy derivative for phase separation.
    """
    return 2.0 * c * (1.0 - c)**2 - 2.0 * c**2 * (1.0 - c)

# Laplacian function using finite difference method
def laplacian(field, dx, dy):
    """
    Computes the Laplacian of a field using finite difference method.
    """
    laplace = (
                      np.roll(field, -1, axis=0) + np.roll(field, 1, axis=0) +
                      np.roll(field, -1, axis=1) + np.roll(field, 1, axis=1) -
                      4.0 * field
              ) / (dx * dy)
    return laplace


# Time integration function (Cahn-Hilliard equation step)
def cahn_hilliard_step(con, mobility, grad_coef, dt, dx, dy):
    """
    Takes one step of the Cahn-Hilliard equation for the concentration field.
    The Cahn-Hilliard equation: dc/dt = M∇²(dF/dc - κ∇²c)
    """
    mu = dfdc(con) - grad_coef * laplacian(con, dx, dy)  # Chemical potential
    dcon_dt = mobility * laplacian(mu, dx, dy)  # Evolution equation
    con = np.clip(con + dcon_dt * dt, 0.00000001, 0.999999999)
    return con


def calculate_total_energy(con, grad_coef, dx, dy):
    """
    Calculates the total energy of the system based on the concentration field.
    """
    energy = 0.0
    Nx, Ny = con.shape
    for i in range(Nx - 1):
        ip = i + 1
        for j in range(Ny - 1):
            jp = j + 1
            energy += (con[i, j]**2 * (1.0 - con[i, j])**2 +
                       0.5 * grad_coef * ((con[ip, j] - con[i, j])**2 +
                                          (con[i, jp] - con[i, j])**2))
    return energy * dx * dy

# Function to save concentration field as VTK file
def save_vtk(save_dir, istep, data1):
    """
    Saves the concentration field at a given timestep as a VTK file.
    """
    fname = os.path.join(save_dir, f"time_{istep}.vtk")
    with open(fname, 'w') as out:
        nz = 1
        npoin = data1.shape[0] * data1.shape[1] * nz
        out.write('# vtk DataFile Version 2.0\n')
        out.write(f'time_{istep}.vtk\n')
        out.write('ASCII\n')
        out.write('DATASET STRUCTURED_GRID\n')
        out.write(f'DIMENSIONS {data1.shape[0]} {data1.shape[1]} {nz}\n')
        out.write(f'POINTS {npoin} float\n')

        for i in range(data1.shape[0]):
            for j in range(data1.shape[1]):
                x = (i - 1) * 1.0
                y = (j - 1) * 1.0
                z = 0.0
                out.write(f'{x:14.6e} {y:14.6e} {z:14.6e}\n')

        out.write(f'POINT_DATA {npoin}\n')
        out.write('SCALARS CON float 1\n')
        out.write('LOOKUP_TABLE default\n')

        for i in range(data1.shape[0]):
            for j in range(data1.shape[1]):
                out.write(f'{data1[i, j]:14.6e}\n')

    print(f"VTK file '{fname}' has been written.")


# Function to create folder for saving results
def create_folder(c0, mobility, grad_coef, batch_name):
    """Create folder for saving results with batch organization"""
    base_dir = os.path.join("D:/sample", batch_name)
    os.makedirs(base_dir, exist_ok=True)
    
    folder_name = f"{c0},{mobility},{grad_coef}"
    target_dir = os.path.join(base_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir

def main_simulation(batch_name, Nx=128, Ny=128, nstep=30000, nprint=200, dx=1.0, dy=1.0):
    """
    Main simulation loop for spinodal decomposition with randomized parameters.
    Saves results in specified batch folder.
    """
    # Run 100 simulations
    for sim in range(100):
        # Randomized parameters within effective ranges
        c0 = random.uniform(0.3, 0.7)  # Concentration near critical point
        mobility = random.uniform(1, 2)  # Moderate mobility
        grad_coef = random.uniform(0.5, 1)  # Balanced gradient coefficient
        noise_amp = random.uniform(0.05, 0.15)  # Sufficient noise to trigger decomposition
        dt = 0.001  # Stable timestep
        
        # Initialize system with random noise
        con = micro_ch_pre(Nx, Ny, c0, iflag=1, noise=noise_amp)
        
        # Create save directory and save parameters
        save_dir = create_folder(c0, mobility, grad_coef, batch_name)
        print(f"\nSimulation {sim+1}/100")
        print(f"Parameters: c0={c0}, mobility={mobility}, grad_coef={grad_coef}")
        
        # Save parameters for training data
        param_file = os.path.join(save_dir, 'parameters.txt')
        with open(param_file, 'w') as f:
            f.write(f'initial_concentration: {c0}\n')
            f.write(f'mobility: {mobility}\n')
            f.write(f'gradient_coefficient: {grad_coef}\n')
            f.write(f'noise_amplitude: {noise_amp}\n')
            f.write(f'timestep: {dt}\n')
        
        # Evolution loop
        for istep in range(nstep + 1):
            con = cahn_hilliard_step(con, mobility, grad_coef, dt, dx, dy)
            
            if istep % nprint == 0:
                energy = calculate_total_energy(con, grad_coef, dx, dy)
                print(f"Step {istep}, Energy: {energy:.6e}")
                save_vtk(save_dir, istep, con)

if __name__ == "__main__":
    # Run with batch name "1"
    main_simulation("57")
