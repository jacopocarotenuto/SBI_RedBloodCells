

############## LIBRARIES ##############

from numba import jit
from numpy import zeros, arange, uint8, int32, float32, sqrt, uint32, ones, vstack, concatenate
from numpy import int64, mean, ceil, where, log2, max, min, median, var, log, array, sum
from numpy.random import randn, uniform
from numpy.fft import fft, ifft, fftfreq
from torch import Tensor
from sbi import utils as utils
from scipy.integrate import cumulative_trapezoid
from scipy.signal import welch
import torch
import time
from time import gmtime, strftime
import numpy as np

############## SIMULATOR AND TIME VARIABLES ##############

@jit(nopython = True)
def Simulator_noGPU(dt, DeltaT, TotalT, theta, i_state = None):
    
    
    time_steps_amount = int64(TotalT/dt) # Number of steps
    sampled_point_amount = int64(TotalT/DeltaT) # Number of sampled points
    sampling_delta_time_steps = int64(DeltaT/dt) # Number of steps between samples
    
    n_sim = theta[0].shape[0]
    
    # Unpack Parameters
    mu_x = theta[0]
    mu_y = theta[1]
    k_x = theta[2]
    k_y = theta[3]
    k_int = theta[4]
    tau = theta[5]
    eps = theta[6]
    D_x = theta[7]
    D_y = theta[8]
    
    
    if len(set([x.shape for x in theta])) != 1:
        raise Exception("Parameters dimension are not all equal. Detected number of different parameters: ", n_sim)
    
    
    # Handle initial state
    if i_state is None:
        x = zeros((n_sim,1), dtype = float32)
        y = zeros((n_sim,1), dtype = float32)
        f = zeros((n_sim,1), dtype = float32)
    else:
        if x.shape == (n_sim,1) or y.shape == (n_sim,1) or f.shape == (n_sim,1):
            raise Exception("Initial state has wrong shape, each should be (n_sim,1)")
        x, y, f = i_state
    
    # Initialize x_trace array
    x_trace = zeros((n_sim, sampled_point_amount), dtype = float32)
    f_trace = zeros((n_sim, sampled_point_amount), dtype = float32)
    y_trace = zeros((n_sim, sampled_point_amount), dtype = float32)

    sampling_counter = int64(1)
    
    # POSSIBLE OPTIM: You could overwrite the last used row of the trace to save memory and not create a proxy array
    
    # TRADEOFF: Memory vs Speed. We can generate the numbers here, or before. Maybe the time is the same... Should test
    
    # CHECK: Benchmark the version with the explicit dx, dy, df and the one with the x, y, f arrays with the calculation in the assigment
    
    for t in arange(time_steps_amount - 1):
        x[:,] = x[:,] + mu_x*(- k_x * x[:,] + k_int*y[:,])*dt                +          sqrt(2*mu_x*D_x*dt)  *   np.random.randn(n_sim,1)
        y[:,] = y[:,] + mu_y*(-k_y*y[:,] + k_int*x[:,] + f[:,])*dt        +          sqrt(2*mu_y*D_y*dt)     *   np.random.randn(n_sim,1)
        f[:,] = f[:,] + -(f[:,]/tau)*dt                                   +          sqrt(2*eps**2*dt/tau)   *   np.random.randn(n_sim,1)

        sampling_counter = sampling_counter + 1
        if sampling_counter == sampling_delta_time_steps:
            x_trace[:, int(t/sampling_delta_time_steps)] = x[:,0]
            f_trace[:, int(t/sampling_delta_time_steps)] = f[:,0]
            y_trace[:, int(t/sampling_delta_time_steps)] = y[:,0]

            sampling_counter = int64(1)

    return x_trace, f_trace, y_trace # Check if this is right

def CheckParameters(dt, DeltaT, TotalT, theta):
    '''
    Checks the variables and parameters for the simulation.

    INPUT
    dt: integration time
    DeltaT: sampling time
    TotalT: total simulation time
    theta: parameters
    '''
    time_steps_amount = int64(TotalT/dt) # Number of steps
    sampled_point_amount = int64(TotalT/DeltaT) # Number of sampled points
    sampling_delta_time_steps = int64(DeltaT/dt) # Number of steps between samples
    n_sim = theta[0].shape[0]
    # Aggiugnere controllo sul TotalT effettivo a fine simulazione
    # Aggiungere controllo sul sampling_delta_time_steps per sanity check
    # Controllare che sampled_point_amount*sampling_delta_time_steps = time_steps_amount
    
    print(f"Your Integration Time (dt) is {dt:.2E} seconds")
    print(f"Your Sampling Time (Delta) is {DeltaT:.2E} seconds, corresponding to a {1/DeltaT:.2f}Hz sampling frequency")
    print(f"Your Total Simulation Time is {TotalT:.2E} seconds")
    print(f"Your Number of Simulated Trajectories is {n_sim:.2E}")
    print(f"The amount of total time steps is {time_steps_amount:.2E}")
    print(f"The amount of sampled points is {sampled_point_amount:.2E}")
    print(f"The gap between two sampled points is {sampling_delta_time_steps:.1E} time steps")
    passed_sanity_checks = True
    print("---- SANITY CHECKS ----")
    if TotalT != DeltaT*sampled_point_amount:
        print(f"WARNING: TotalT is {TotalT}s, but DeltaT*sampled_point_amount is {DeltaT*sampled_point_amount}s")
        passed_sanity_checks = False
    if sampled_point_amount*sampling_delta_time_steps != time_steps_amount:
        print(f"WARNING: sampled_point_amount*sampling_delta_time_steps is {sampled_point_amount*sampling_delta_time_steps}, but time_steps_amount is {time_steps_amount}")
        passed_sanity_checks = False
    if time_steps_amount*dt != TotalT:
        print(f"WARNING: time_steps_amount*dt is {time_steps_amount*dt}, but TotalT is {TotalT}")
        passed_sanity_checks = False
    if dt*sampling_delta_time_steps != DeltaT:
        print(f"WARNING: dt*sampling_delta_time_steps is {dt*sampling_delta_time_steps}, but DeltaT is {DeltaT}")
        passed_sanity_checks = False
    if len(set([x.shape for x in theta])) != 1:
        raise Exception("Parameters dimension are not all equal. Detected number of different parameters: ", n_sim)
    if passed_sanity_checks:
        print("All checks passed")
        
    return None

def compute_entropy_production(x_trace, y_trace, f_trace, theta, n_sim):
    '''
    Compute the entropy production for the given traces and parameters
    
    INPUT 
    x_trace: array of shape (n_sim, sampled_point_amount) with the x traces
    y_trace: array of shape (n_sim, sampled_point_amount) with the y traces
    f_trace: array of shape (n_sim, sampled_point_amount) with the f traces
    theta: array of shape (9, n_sim) with the parameters

    OUTPUT
    S_mean: mean entropy production
    Fx: array of shape (n_sim, sampled_point_amount) with the x forces
    Fy: array of shape (n_sim, sampled_point_amount) with the y forces
    S_tot: array of shape (n_sim) with the entropy production for each simulation
    '''

    Fx = []
    Fy = []
    S_tot = []

    for i in range (n_sim):
        # Unpack Parameters
        mu_x = theta[0][i]
        mu_y = theta[1][i]
        k_x = theta[2][i]
        k_y = theta[3][i]
        k_int = theta[4][i]
        tau = theta[5][i]
        eps = theta[6][i]
        D_x = theta[7][i]
        D_y = theta[8][i]

        

        x, y, f = x_trace[i], y_trace[i], f_trace[i]

        # Compute the force
        F_x = - k_x * x + k_int * y
        F_y = - k_y * y + k_int * x + f
        Fx.append(F_x)
        Fy.append(F_y)

        # Compute the entropy production
        S_x = sum((x_trace[i][1:] - x_trace[i][:-1]) * F_x[:-1] / D_x)
        S_y = sum((f_trace[i][1:] - f_trace[i][:-1]) * F_y[:-1] / D_y)
        S = S_x + S_y
        S_tot.append(S)
    
    S_mean = mean(S_tot)
    return S_mean, array(Fx), array(Fy), array(S_tot)

############## CORRELATION ##############



############## INTEGRATION AND SUMMARY STATISTICS ##############


def get_theta_from_prior(prior_limits, n_sim):
    '''
    Get parameters drawn from the prior.

    INPUT
    prior_limits: prior limits
    n_sim: number of simulated trajectories

    OUTPUT
    theta: parameters drawn from the prior
    theta_torch: parameters drawn from the prior in torch format
    prior_box: torch prior box distribution
    '''
    # Get parameters drawn from the prior
    theta = [uniform(prior_limits[i][0], prior_limits[i][1], size=(n_sim, 1)) for i in range(len(prior_limits))]
    theta_torch = torch.from_numpy(array(theta)[:,:,0].T).to(torch.float32)

    # Get the torch prior box (used in sbi)
    prior_limits = array(prior_limits)
    prior_box = utils.torchutils.BoxUniform(low=torch.tensor(prior_limits[:, 0]), high=torch.tensor(prior_limits[:, 1]))
    
    return theta, theta_torch, prior_box










# def get_summary_statistics(list_stat, x_trace, f_trace, theta, DeltaT, k_psd, t, t_corr):
#     '''
#     Selects the summary statistics to compute.

#     INPUT
#     list_stat: list of summary statistics
#     x_trace: x trace signal
#     f_trace: f trace signal
#     theta: parameters
#     DeltaT: sampling time
#     k_psd: number of segments to divide the signal
#     t: time array
#     t_corr: maximum time for the correlation

#     OUTPUT
#     summary: summary statistics
#     '''
#     n_sim = x_trace.shape[0]
#     sampled_point_amount = x_trace.shape[1]
#     theta_numpy = array(theta)
#     Sample_frequency = 1/DeltaT

#     for i in range(n_sim):
#         single_x_trace = x_trace[i]
#         single_f_trace = f_trace[i]
#         theta_i = theta_numpy[:, i]

#         summary_i = []

#         for stat in list_stat: 
#             corr_dependency = False
#             psdx_dependency = False
#             psdf_dependency = False
            
#             if stat == "Cxx" or stat == "Cfx" or stat == "Cff":
#                 Cxx, Cfx, Cff, idx_corr = stat_corr(single_x_trace, single_f_trace, DeltaT, t, t_corr)  
#                 corr_dependency = True

#                 if stat == "Cxx": summary_i.append(Cxx[idx_corr])
#                 if stat == "Cfx": summary_i.append(Cfx[idx_corr])
#                 if stat == "Cff": summary_i.append(Cff[idx_corr])
            
#             if stat == "s_redx":
#                 if corr_dependency == False:
#                     Cxx, Cfx, Cff, idx_corr = stat_corr(single_x_trace, single_f_trace, DeltaT, t, t_corr)
#                     corr_dependency = True
#                 S_redx = stat_s_redx(Cxx, t_corr, t, theta_i)
#                 summary_i.append(S_redx)
            
#             if stat == "s_redf":
#                 if corr_dependency == False:
#                     Cxx, Cfx, Cff, idx_corr = stat_corr(single_x_trace, single_f_trace, DeltaT, t, t_corr)
#                     corr_dependency = True
#                 S_redf = stat_s_redf(Cfx, t_corr, t, theta_i)
#                 summary_i.append(S_redf)

#             if stat == "psdx":
#                 psdx = stat_psd(single_x_trace, k_psd, Sample_frequency, sampled_point_amount)
#                 psdx_dependency = True
#                 summary_i.append(psdx)

#             if stat == "psdf":
#                 psdf = stat_psd(single_f_trace, k_psd, Sample_frequency, sampled_point_amount)
#                 psdf_dependency = True
#                 summary_i.append(psdf)

#             if stat == "ts_psdx":
#                 if psdx_dependency == False:
#                     psdx = stat_psd(single_x_trace, k_psd, Sample_frequency, sampled_point_amount)
#                     psdx_dependency = True
#                 ts_psdx = stat_timeseries(psdx)
#                 summary_i.append(ts_psdx)
            
#             if stat == "ts_psdf":
#                 if psdf_dependency == True:
#                     psdf = stat_psd(single_f_trace, k_psd, Sample_frequency, sampled_point_amount)
#                     psdf_dependency = False
#                 ts_psdf = stat_timeseries(psdf)
#                 summary_i.append(ts_psdf)

#             if stat == "hermite":
#                 hermite_stat = stat_hermite(single_x_trace, [1])
#                 summary_i.append(hermite_stat)
#                 hermite_stat = stat_hermite(single_x_trace, [0,0,1])
#                 summary_i.append(hermite_stat)
#                 hermite_stat = stat_hermite(single_x_trace, [0,0,0,0,1])
#                 summary_i.append(hermite_stat)

#         summary_i = concatenate(summary_i, axis=0)
        
#         if i == 0:
#             summary = summary_i.copy()
#         else:
#             summary = vstack((summary, summary_i))
    
#     summary = torch.from_numpy(array(summary)).to(torch.float32)
#     return summary