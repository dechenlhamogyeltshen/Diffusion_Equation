# Numerical schemes for simulating diffusion for outer codes diffusion.py and\
# numerical_analysis.py

import numpy as np
# The linear algebra package for BTCS (for solving the matrix equation)
import numpy.linalg as la

def FTCS_fixed_zeroGrad(phi, K, Q, dx, dt, nt):
    ''' 
    Diffuses the initial condition phi with diffusion coefficient K, source 
    term Q, grid spacing dx, time step dt for nt time steps. Fixed boundary 
    conditions at phi[0] and zero gradient at phi[-1].
    Diffusion is calculated using the FTCS scheme to solve 
    dphi/dt = K d^2phi/dx^2 + Q
    
    Parameters:
    phi : 1darray. Profile to diffuse
    K : float. The diffusion coefficient
    Q : float. The source term
    dx : float. The grid spacing
    dt : float. The time step
    nt : int. The number of time steps
    
    Returns:
    phi : 1darray. phi after solution of the diffusion equation
    '''
    
    nx = len(phi)
    d = K*dt/dx**2 # Non-dimensional diffusion coefficient
    
    # New time-step array for phi
    phiNew = phi.copy()
    
    # FTCS for all time steps
    for it in range(nt):
        # Loop over all internal points
        for ix in range(1,nx-1):
            phiNew[ix] = phi[ix] + d*(phi[ix+1] - 2*phi[ix] + phi[ix-1]) + dt*Q
        
        # Apply boundary conditions (fixed value at i=0, fixed gradient at end)
        phiNew[0] = 293.
        phiNew[nx-1] = phiNew[nx-2]
        
        # Update phi for next time-step
        phi = phiNew.copy()
        
    return phi

def BTCS_fixed_zeroGrad(phi, K, Q, dx, dt, nt):
    '''
    Diffuses the initial condition phi with diffusion coefficient K, source 
    term Q, grid spacing dx, time step dt for nt time steps. Fixed boundary 
    conditions at phi[0] and zero gradient at phi[-1].
    Diffusion is calculated using the BTCS scheme to solve
    dphi/dt = K d^2phi/dx^2 + Q

    Parameters:
    phi : 1darray. Profile to diffuse
    K : float. The diffusion coefficient
    Q : float. The source item
    dx : float. The grid spcaing
    dt : float. The time step
    nt : int. The number of time steps

    Returns:
        phi : 1darray. phi after solution of the diffusion equation
    '''
    
    nx = len(phi)
    
    # Non-dimensional diffusion coefficient
    d = dt*K/dx**2
    
    # Array representing BTCS
    M = np.zeros([nx,nx])
    # Fixed value boundary conditions at the start
    M[0,0] = 1
    # Zero gradient boundary conditions at the end
    M[-1,-1] = 1
    M[-1,-2] = -1
    # Other array elements
    for i in range(1,nx-1):
        M[i,i-1] = -d
        M[i,i] = 1+2*d
        M[i,i+1] = -d
        
    # BTCS for all time steps
    for it in range(nt):
        # RHS vector
        RHS = phi + dt*Q
        # RHS for fixed value boundary conditions at start
        RHS[0] = phi[0]
        # RHS for zero gradient boundary conditions at end
        RHS[-1] = 0
        
        # Solve the matrix equaiton to update phi
        phi = la.solve(M,RHS)
        
    return phi

def FTCS_periodic(phi,K,dx,dt,nt):
    '''
    Diffuses the initial condition phi with diffusion coefficient K, grid 
    spacing dx, time step dt for nt time steps.
    Diffusion is calculated using the FTCS scheme to solve 
    dphi/dt = K d^2phi/dx^2

    Parameters
    ----------
    phi : 1darray. Profile to diffuse.
    K : float. The diffusion coefficient.
    dx : float. THe grid spacing.
    dt : float. The time step.
    nt : int. The number of time steps.

    Returns
    -------
    phi: 1darray. phi after solution of the diffusion equation

    '''
    
    nx = len(phi)
    d = K*dt/dx**2 # Non-dimensional diffusion coefficient
    
    # New time-step array for phi
    phiNew = phi.copy()
    
    # FTCS for all time steps
    for it in range(nt):
        # Loop over all internal points
        for ix in range(1,nx-1):
            phiNew[ix] = phi[ix] + d*(phi[ix+1] - 2*phi[ix] + phi[ix-1])
            
        # Apply periodic boundary conditions
        phiNew[0] = phi[0] + d*(phi[nx-2] - 2*phi[0] + phi[1])
        phiNew[nx-1] = phiNew[0]
        
        # Update phi for next time-step
        phi = phiNew.copy()
        
    return phi


            