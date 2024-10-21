# Outer code for comparing analytic and FTCS solutions of the diffusion 
# equation and test convergence

import matplotlib.pyplot as plt
import numpy as np
import math

# Read in all the schemes, initial conditions and other helper code
from diffusionSchemes import *

def compare():
    """ Solve the diffusion equation using FTCS and analytically"""
    # Parameters
    xmin = 0.           # Start of model domain 
    xmax = 2*math.pi    # End of model domain
    nx = 20             # Number of grid points, including both ends
    endTime = 9.0         # Number of seconds of the whole simulation
    nt = 1000           # Number of time steps taken to get to the endTime
    K = 1.              # The diffusion coefficient

    # Other derived parameters
    dt = endTime/nt         # The time step
    dx = (xmax - xmin)/(nx - 1)     # The grid spacing
    d = K*dt/dx**2                  # Non-dimensional diffusion coefficient
    print("dx =", dx, "dt =", dt, "non-dimensional diffusion coefficient =", d)
    
    # Spatial points
    x = np.linspace(xmin,xmax,nx)
    # Initial condition
    phi = np.cos(2*x)

    # Diffusion using FTCS and BTCS
    phi_FTCS = FTCS_periodic(phi.copy(), K, dx, dt, nt)
    phi_Ana = math.exp(-K*4*endTime)*phi

    # Plot the solution
    plt.rcParams["font.size"] = 14
    plt.plot(x, phi_FTCS, label='FTCS', color='blue', marker='+')
    plt.plot(x, phi_Ana, label='Analytic', color='red', linestyle='dashed')
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('phi')
    plt.savefig('plots/FTCS_Ana.pdf')
    plt.show()

def convergence():
    """ Solve the diffusion equation and calculate l2 error norms for different
    time and spatial steps"""
    # Parameters
    xmin = 0.           # Start of model domain
    xmax = 2*math.pi    # End of model domain
    endTime = 9.0         # Number of seconds of the whole simulation
    K = 1.              # The diffusion coefficient
    
    # Empty arrays to save values of dx and l2 errors
    l2_norms = np.zeros(5)
    dxs = np.zeros(5)
    
    # Different number of grid points, including both ends
    nxs = [5, 9, 17, 33, 65]
    # Corresponding time steps (number of time steps taken to get to endTime)
    nts = [100, 400, 1600, 6400, 25600]         
    
    # Specify plot font size
    plt.rcParams["font.size"] = 10
    
    # Loop through different nx, nt configurations and plot the l2 error norms
    for i, (nx, nt) in enumerate(zip(nxs, nts)):
        # Saptial points
        x = np.linspace(xmin,xmax,nx)
        # Initial condition
        phi = np.cos(2*x)
        
        # Derived parameters
        dt = endTime/nt                 # The time step
        dx = (xmax - xmin)/(nx - 1)     # The grid spacing
        dxs[i] = dx
        d = K*dt/dx**2                  # Non-dimensional diffusion coefficient
        print(f"Iteration {i+1}: endTime = {endTime}, dx = {dx}, nt = {nt}, \
              dt = {dt}, non-dimensional coefficient = {d}")

        # Diffusion using FTCS with periodic boundary conditions
        phi_FTCS = FTCS_periodic(phi.copy(), K, dx, dt, nt)
        phi_Ana = math.exp(-K*4*endTime)*phi
        
        # Calculate l2 error norm
        error = phi_FTCS - phi_Ana
        l2_norm = math.sqrt(np.sum(dx*np.square(error))) / math.sqrt(\
                                                    np.sum(dx*np.square(phi_Ana)))
        l2_norms[i] = l2_norm
        
    # Plot the l2 error norms
    plt.loglog(dxs, l2_norms, label='FTCS L2 error norm')
    plt.xlabel('dx')
    plt.ylabel('l2')
    plt.savefig('plots/FTCS_convergence.pdf', bbox_inches='tight')
    plt.show()
    
compare()
convergence()
