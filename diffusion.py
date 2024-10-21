# Outer code for setting up the diffusion problem, calculate and plot.

import matplotlib.pyplot as plt
import numpy as np

# Read in all the schemes, initial conditions and other helper code
from diffusionSchemes import *

### The main code is inside a funtion to avoid global variables
def steady_state():
    """ Solve the diffusion equation with different numbers of timesteps """
    
    # Parameters
    zmin = 0.           # Start of model domain (m)
    zmax = 1e3          # End of model domain (m)
    nz = 21             # Number of grid points, including both ends
    K = 1.              # The diffusion coefficient (m^2/s)
    Tinit = 293.        # The initial conditions
    Q = -1.5/86400      # The healing rate
    
    # Different end times (number of seconds of the whole simulation)
    endTimes = [4.8e5, 7.2e5, 9.6e5, 1.2e6, 1.44e6, 1.68e6, 1.92e6]  
    # Corresponding time steps (number of time steps taken to get to endTime)
    nts = [800, 1200, 1600, 2000, 2400, 2800, 3200]         
    
    # Derived parameters
    dz = (zmax - zmin)/(nz - 1)     # The grid spacing
    print("dz =", dz)
    
    # Height points
    z = np.linspace(zmin,zmax,nz)
    # Initial condition
    T = Tinit * np.ones(nz)
    
    # Specify plot font size
    plt.rcParams["font.size"] = 10
    
    # Loop through different endTime and nt configurations and plot solutions
    for i, (endTime, nt) in enumerate(zip(endTimes, nts)):
        dt = endTime/nt           # The time step
        d = K*dt/dz**2          # Non-dimensional diffusion coefficient
        print(f"Iteration {i+1}: endTime = {endTime}, nt = {nt}, dt = {dt},\
              non-dimensional coefficient = {d}")

        # Diffusion using FTCS and BTCS
        T_FTCS = FTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt)
        T_BTCS = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt)

        # Plot the solution for each iteration
        plt.plot(T_FTCS - Tinit, z, label=f'FTCS ({endTime} s)', marker='+')
        plt.plot(T_BTCS - Tinit, z, label=f'BTCS', linestyle='dashed')
    
    # Plot specifications
    plt.ylim([0, 1000])
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.xlabel('$T-T_0$ (K)')
    plt.ylabel('z (m)')
    plt.savefig('plots/FTCS_BTCS_steady_state.pdf', bbox_inches='tight')
    plt.show()

def stability():
    """ Solve the diffusion equation with different time steps """
    
    # Parameters
    zmin = 0.           # Start of model domain (m)
    zmax = 1e3          # End of model domain (m)
    nz = 21             # Number of grid points, including both ends
    endTime = 1.92e6    # Number of seconds of the whole simulation
    K = 1.              # The diffusion coefficient (m^2/s)
    Tinit = 293.        # The initial conditions
    Q = -1.5/86400      # The healing rate
            
    # Number of time steps for unstable FTCS scheme
    ntu = 1500
    
    # One time step for BTCS scheme
    nt = 1
    
    # Derived parameters
    dtu = endTime/ntu   # The time step for unstable
    dt = endTime/nt     # The time step for one step
    dz = (zmax - zmin)/(nz - 1)     # The grid spacing
    du = K*dtu/dz**2    # Non-dimensional diffusion coefficient for unstable 
    d = K*dt/dz**2     # Non-dimensional diffusion coefficient for one step
    print("For an unstable FTCS, dz =", dz, "dt =", dtu, "non-dimensional\
          diffusion coefficient =", du)
    print("For an one time step, dz =", dz, "dt =", dt, "non-dimensional\
          diffusion coefficient =", d)
          
    # Height points
    z = np.linspace(zmin,zmax,nz)
    # Initial condition
    T = Tinit * np.ones(nz)
    
    # Solve and plot solutions for unstable FTCS scheme
    T_FTCS_u = FTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dtu, ntu)
    T_BTCS_u = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dtu, ntu)
    
    plt.rcParams["font.size"] = 10
    plt.plot(T_FTCS_u - Tinit, z, label='FTCS', marker='+')
    plt.plot(T_BTCS_u - Tinit, z, label='BTCS', linestyle='dashed')
    
    plt.ylim([0,1000])
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.xlabel('$T-T_0$ (K)')
    plt.ylabel('z (m)')
    plt.savefig('plots/FTCS_BTCS_unstable.pdf', bbox_inches='tight')
    plt.show()
    
    # Solve and plot solutions for one time step BTCS scheme
    T_BTCS = BTCS_fixed_zeroGrad(T.copy(), K, Q, dz, dt, nt)
    
    plt.rcParams["font.size"] = 10
    plt.plot(T_BTCS - Tinit, z, label='BTCS', marker='+')
    plt.ylim([0,1000])
    plt.legend(loc='best', bbox_to_anchor=(1, 1))
    plt.xlabel('$T-T_0$ (K)')
    plt.ylabel('z (m)')
    plt.savefig('plots/TCS_one_timestep.pdf', bbox_inches='tight')
    plt.show()
    
steady_state()
stability()
