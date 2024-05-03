from numba import jit
import numpy as np

@jit(nopython=True)
def ComputeTheoreticalEntropy(theta):
    '''
    Compute the entropy production for the given parameters.

    INPUT
    theta: list of 9 arrays of parameters. The 9 arrays must be all of the same length.

    OUTPUT
    sigmas: entropy production for each simulation
    sigma_mean: mean entropy production
    '''
    
    if len(theta) != 9:
        raise ValueError('There must be 9 parameters in theta')

    if len(set([x.shape for x in theta])) != 1:
        raise Exception("Parameters dimension are not all equal.")
    
    n_sim = theta[0].shape[0]
    sigmas = np.zeros((n_sim,1), dtype = np.float64)
    
    for i in range(n_sim):
        mu_x = theta[0][i]
        mu_y = theta[1][i]
        k_x = theta[2][i]
        k_y = theta[3][i]
        k_int = theta[4][i]
        tau = theta[5][i]
        eps = theta[7][i]

        sigma = (mu_y * eps**2) / ((1 + k_y * mu_y * tau) - ((k_int ** 2 * mu_x * mu_y * tau ** 2) / (1 + k_x * mu_x * tau)))
        sigmas[i] = sigma

    sigma_mean = np.float64(np.mean(sigmas))

    return sigmas, sigma_mean