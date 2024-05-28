from numba import jit
from numpy import zeros, arange, uint8, int32, float32, sqrt, uint32, ones, vstack, concatenate
from numpy import int64, mean, ceil, where, log2, max, min, median, var, log, array, sum
from numpy.random import randn, uniform
from numpy.fft import fft, ifft, fftfreq
from sbi import utils as utils
from scipy.integrate import cumulative_trapezoid
from scipy.signal import welch
import numpy as np
import torch
import os
import _pickle as pickle
from scipy.optimize import curve_fit
from scipy.stats import zscore


def get_theta_from_prior(prior_limits, n_sim):
    '''
    Get parameters drawn from the prior.

    INPUT
    prior_limits: prior limits
    n_sim: number of simulated trajectories

    OUTPUT
    theta: parameters drawn from the prior
    theta_torch: parameters drawn from the prior in torch format
    '''

    # Get parameters drawn from the prior
    theta = [np.random.uniform(prior_limits[i][0], prior_limits[i][1], size=(n_sim, 1)) for i in prior_limits]
    theta_numpy = np.array(theta)
    theta_torch = torch.from_numpy(theta_numpy[:, :, 0]).to(torch.float32)

    return theta_numpy, theta_torch


def get_prior_limit_list(prior_limits):
    prior_limits_list = [[prior_limits[i][0], prior_limits[i][1]] for i in prior_limits]
    return prior_limits_list


def get_prior_box(prior_limits):
    prior_limits_list = get_prior_limit_list(prior_limits)
    prior_limits_array = array(prior_limits_list)
    prior_box = utils.torchutils.BoxUniform(low=torch.tensor(prior_limits_array[:, 0]), high=torch.tensor(prior_limits_array[:, 1]))
    return prior_box




@jit(nopython=True)
def ComputeTheoreticalEntropy(theta, mu_x=2.8e4, k_x=6e-3, kbT=3.8):
    '''
    Compute the entropy production for the given parameters.

    INPUT
    theta: list of 9 arrays of parameters. The 9 arrays must be all of the same length.

    OUTPUT
    sigmas: entropy production for each simulation
    sigma_mean: mean entropy production
    '''
    D_x = kbT * mu_x
    
    if len(theta) != 5:
        raise ValueError('There must be 5 parameters in theta')

    if len(set([x.shape for x in theta])) != 1:
        raise Exception("Parameters dimension are not all equal.")
    
    n_sim = theta.shape[1]
    sigmas = np.zeros((n_sim,1), dtype = np.float64)
    
    for i in range(n_sim):
        mu_y = theta[0, i]
        k_y = theta[1, i]
        k_int = theta[2, i]
        tau = theta[3, i]
        eps = theta[4, i]

        sigma = (mu_y * eps**2) / ((1 + k_y * mu_y * tau) - ((k_int ** 2 * mu_x * mu_y * tau ** 2) / (1 + k_x * mu_x * tau)))
        sigmas[i] = sigma

    sigma_mean = np.float64(np.mean(sigmas))

    return sigmas, sigma_mean


def ComputeEmpiricalEntropy(x_trace, y_trace, f_trace, theta, n_sim, t, mu_x=2.8e4, k_x=6e-3, kbT=3.8):
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

    D_x = kbT * mu_x
    for i in range (n_sim):
        # Unpack Parameters
        mu_y = theta[0, i]
        k_y = theta[1, i]
        k_int = theta[2, i]
        tau = theta[3, i]
        eps = theta[4, i]

        D_y = kbT * mu_y

        x, y, f = x_trace[i], y_trace[i], f_trace[i]

        # Compute the force
        F_x = - k_x * x + k_int * y
        F_y = - k_y * y + k_int * x + f
        Fx.append(F_x)
        Fy.append(F_y)

        # Compute the entropy production
        S_x = sum((x_trace[i][1:] - x_trace[i][:-1]) * F_x[:-1] / t)
        S_y = sum((y_trace[i][1:] - y_trace[i][:-1]) * F_y[:-1] / t)
        S = S_x + S_y
        S_tot.append(S)
    
    S_mean = mean(S_tot)
    return S_mean, array(Fx), array(Fy), array(S_tot)



def corr(x,y,nmax,dt=False):
    '''
    Performs the cross correlation between two single-input signals x and y.

    INPUT
    x: input signal 1
    y: input signal 2
    nmax: maximum number of lags
    dt: time step (default=False)

    OUTPUT
    corr: cross-correlation between x and y
    '''

    assert len(x)==len(y), 'x and y must have the same length'

    n=len(x)
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**ceil(log2(ext_size)).astype('int')

    xp=x-mean(x)
    yp=y-mean(y)

    # do fft and ifft
    cfx=fft(xp,fsize)
    cfy=fft(yp,fsize)
    if dt != False:
        freq = fftfreq(n, d=dt)
        idx = where((freq<-1/(2*dt))+(freq>1/(2*dt)))[0]
        
        cfx[idx]=0
        cfy[idx]=0
        
    sf=cfx.conjugate()*cfy
    corr=ifft(sf).real
    corr=corr/n

    return corr[:nmax]


def hermite(x, i):
    std_x = np.std(x)
    z = x/std_x
    zeros = np.zeros(30)
    index = zeros
    index[i] = 1
    return np.mean(((np.exp(-z**2/2)*np.polynomial.hermite.hermval(z, index.tolist())*(2**i*
            np.math.factorial(i)*np.sqrt(np.pi))**-0.5) /np.sqrt(std_x)))


def stat_corr(single_x_trace, single_f_trace, DeltaT, t, t_corr):
    '''
    Computes the autocorrelation and cross-correlation for a single x and f trace signal.

    INPUT
    singles_x_trace: single x trace signal
    singles_f_trace: single f trace signal
    DeltaT: sampling time
    t: time array
    t_corr: maximum time for the correlation

    OUTPUT
    Cxx: autocorrelation x signal
    Cfx: cross-correlation xf signal
    Cff: autocorrelation f signal
    '''

    sampled_point_amount = single_x_trace.shape[0]
    idx_corr = where((t>0)*(t<t_corr))[0]
    Cxx= corr(single_x_trace, single_x_trace, sampled_point_amount, dt=DeltaT) # compute the autocorrellation for each x trace
    Cfx = corr(single_f_trace, single_x_trace, sampled_point_amount, dt=DeltaT) # compute the cross-correllation for each x and f trace
    Cff = corr(single_f_trace, single_f_trace, sampled_point_amount, dt=DeltaT) # compute the autocorrellation for each f trace    

    return Cxx, Cfx, Cff


def stat_corr_single(single_x_trace, DeltaT):
    '''
    Computes the autocorrelation for a single x trace signal.

    INPUT
    singles_x_trace: single x trace signal
    DeltaT: sampling time
    ((t: time array
    t_corr: maximum time for the correlation))

    OUTPUT
    Cxx: autocorrelation x signal
    '''

    sampled_point_amount = single_x_trace.shape[0]
    #idx_corr = where((t>0)*(t<t_corr))[0]
    Cxx= corr(single_x_trace, single_x_trace, sampled_point_amount, dt=DeltaT) # compute the autocorrellation for each x trace

    return Cxx


def stat_s_redx(Cxx, t_corr, t, mu_x=2.8e4, k_x=6e-3, kbT=3.8):
    '''
    Computes the reduced energy production for a single x trace signal.

    INPUT
    Cxx: autocorrelation signal
    t_corr: maximum time for the correlation
    t: time array
    theta_i: parameters

    OUTPUT
    S_red: reduced x energy production
    '''
    D_x = kbT * mu_x
    
    S1 = cumulative_trapezoid(Cxx, x=t, axis=-1, initial=0)
    S1 = cumulative_trapezoid(S1, x=t, axis=-1, initial=0)
    idx_corr = where((t>0)*(t<t_corr))[0]
    S_red1 = (Cxx[0]-Cxx[idx_corr])/(D_x*t[idx_corr]) # First term in S_red
    S_red2 = ((mu_x*k_x)**2)*S1[idx_corr]/(D_x*t[idx_corr]) # Second term in S_red
    S_red = S_red1 + S_red2 # Compute S_red

    return S_red1, S_red2, S_red


def stat_s_redf(Cfx, t_corr, t, mu_x=2.8e4, k_x=6e-3, kbT=3.8):
    '''
    Computes the reduced energy production for a xf trace signal.

    INPUT
    Cxx: autocorrelation signal
    t_corr: maximum time for the correlation
    t: time array
    theta_i: parameters

    OUTPUT
    S_red: reduced f energy production
    '''
    D_x = kbT * mu_x

    idx_corr = where((t>0)*(t<t_corr))[0]
    S2f = cumulative_trapezoid(Cfx - Cfx[0], x=t, axis=-1, initial=0)
    S3f = cumulative_trapezoid(Cfx, x=t, axis=-1, initial=0)
    S3f = -mu_x*k_x*cumulative_trapezoid(S3f, x=t, axis=-1, initial=0)
    S_redf = 1-(S2f[idx_corr]+S3f[idx_corr])/(D_x*t[idx_corr]) # the energy production is to to the fluctuation-dissipation theorem
    
    return S_redf
  

def stat_psd(single_trace, nperseg, Sample_frequency):
    '''
    Computes the power spectral density for a single trace signal.

    INPUT
    single_trace: single trace signal
    k: number of segments to divide the signal
    Sample_frequency: sampling frequency
    sampled_point_amount: number of sampled points

    OUTPUT
    psd: power spectral density
    '''
    frequencies, psd = welch(single_trace, fs=Sample_frequency, nperseg=nperseg)
    return psd


def stat_psd_mean(single_trace, nperseg, Sample_frequency):
    _ , psd = welch(single_trace, fs=Sample_frequency, nperseg=nperseg)
    return np.array([mean(psd), np.std(psd)])


def stat_timeseries(single_timeseries):
    '''
    Computes the summary statistics for a single time series signal.

    INPUT
    single_timeseries: single time series signal

    OUTPUT
    s: summary statistics
    '''
    statistics_functions = [[mean, var, median, max, min], 
                            [lambda x: hermite(x, i) for i in range(1, 6)]]
    # s = zeros((len(statistics_functions)))

    # for j, func in enumerate(statistics_functions):
    #     s[j] = func(single_timeseries)

    results = [func(single_timeseries) for l in statistics_functions for func in l]
    s = np.concatenate([results])
    return s

def stat_hermite(x):
    '''
    Computes the Hermite statistics for a single trace signal.

    INPUT
    x: single trace signal

    OUTPUT
    s: Hermite statistics
    '''
    s = np.array([])
    for i in range(0,13,2):
        s = np.concatenate((s, [hermite(x, i)]))
    return s

def stat_mode(cxx, dt, mean_psd):
    def f(t, a0, a2, a4, a6, a8, a10, a12, a14, a16, a18, a20):
        t = t*mean_psd
        h = hermite
        return np.sqrt(mean_psd)*(a0*h(t, 0) + a2*h(t, 2) + a4*h(t, 4) + a6*h(t, 6) + a8*h(t, 8) + a10*h(t, 10) + a12*h(t, 12)+
                                    a14*h(t, 12) + a16*h(t, 16) + a18*h(t, 18) + a20*h(t, 20))
    tp = np.linspace(0, dt*cxx.shape[0], cxx.shape[0])
    popt, _ = curve_fit(f, tp, cxx)
    return popt

def stat_Tucci(single_x_trace, nperseg, Sample_frequency, cxx, dt, mean_psd):
    x = single_x_trace
    x_std = np.std(x)

    herm = stat_hermite(x)

    psd = stat_psd(x, nperseg, Sample_frequency)
    psd_m = mean(psd)
    psd_std = np.std(psd)

    mode = stat_mode(cxx, dt, mean_psd)

    return np.array([x_std, *herm, psd_m, psd_std, *mode])


def compute_summary_statistics(single_x_trace, single_theta, DeltaT = 1/25e3, TotalT = 10):
    summary_statistics = {}
    t = np.linspace(0., TotalT, single_x_trace.shape[0])
    t_corr = TotalT/20 # Hyperparameter
    
    # Autocorrelation
    Cxx = stat_corr_single(single_x_trace, DeltaT)
    idx_corr = where((t>0)*(t<t_corr))[0]
    cxx = Cxx[idx_corr]
    summary_statistics["Cxx"] = cxx  

    idx_clean_corr = np.linspace(0, len(cxx)-1, 500, dtype=np.int32)
    idx_clean_corr_log = np.logspace(0, np.log10(len(cxx)-1), 20, dtype=np.int32)
    summary_statistics["Cxx_cl_lin"] = cxx[idx_clean_corr]
    summary_statistics["Cxx_cl_log"] = cxx[idx_clean_corr_log]
    
    # S red
    S_red1, S_red2, S_red = stat_s_redx(Cxx, t_corr, t)
    summary_statistics["s_red1"] = S_red1
    summary_statistics["s_red2"] = S_red2
    summary_statistics["s_redx"] = S_red 

    summary_statistics["s_redx_cl_lin"] = S_red[idx_clean_corr]
    summary_statistics["s_redx_cl_log"] = S_red[idx_clean_corr_log]
    
    # Power spectral density
    psdx = stat_psd(single_x_trace, nperseg=1000, Sample_frequency=1/DeltaT)
    summary_statistics["psdx"] = psdx
    
    # Time series of the power spectral density and the x_trace
    summary_statistics["ts_psdx"] = stat_timeseries(psdx)
    summary_statistics["ts_x"] = stat_timeseries(single_x_trace)
    
    # Hermite coefficients
    summary_statistics["hermite"] = stat_hermite(single_x_trace)

    # Cxx decomposition in Hermite Coefficients
    summary_statistics["modes"] = stat_mode(Cxx, 1e-6, mean(psdx))

    # Tucci's summary statistics
    summary_statistics["tucci"] = stat_Tucci(single_x_trace, 1000, 1/DeltaT, cxx, 1e-6, mean(psdx))
 
    # Parameters
    summary_statistics["theta"] = single_theta

    return summary_statistics


def select_summary_statistics(summary_statistics, selected_statistics, z_score=False, cl_lin=-1, cl_log=-1):
    selected_statistics = selected_statistics.copy()
    # Check that the selected statistics are in the summary statistics
    assert set(selected_statistics).issubset(set(summary_statistics.keys()))
    "The selected statistics are not in the summary statistics"

    assert cl_log < 0 or cl_lin < 0, "You cannot subsample bot 'lin' and 'log' at the same time"

    # Post-subselection of Cxx and s_redx
    if cl_lin > 0:
        idx_clean_corr = np.linspace(0, len(summary_statistics["Cxx"])-1, cl_lin, dtype=np.int32)
        if "Cxx" in selected_statistics:
            summary_statistics["Cxx"] = summary_statistics["Cxx"][idx_clean_corr]
        if "s_redx" in selected_statistics:
            summary_statistics["s_redx"] = summary_statistics["s_redx"][idx_clean_corr]
        if "s_red1" in selected_statistics:
            summary_statistics["s_red1"] = summary_statistics["s_red1"][idx_clean_corr]
        if "s_red2" in selected_statistics:
            summary_statistics["s_red2"] = summary_statistics["s_red2"][idx_clean_corr]

    if cl_log > 0:
        idx_clean_corr = np.logspace(0, np.log10(len(summary_statistics["Cxx"])-1), cl_log, dtype=np.int32)
        if "Cxx" in selected_statistics:
            summary_statistics["Cxx"] = summary_statistics["Cxx"][idx_clean_corr]
        if "s_redx" in selected_statistics:
            summary_statistics["s_redx"] = summary_statistics["s_redx"][idx_clean_corr]
        if "s_red1" in selected_statistics:
            summary_statistics["s_red1"] = summary_statistics["s_red1"][idx_clean_corr]
        if "s_red2" in selected_statistics:
            summary_statistics["s_red2"] = summary_statistics["s_red2"][idx_clean_corr]


    # Check if theta is selected for testing reasons
    theta_selected = False
    if "theta" in selected_statistics:
        theta_selected = True
        selected_statistics.remove("theta")

    # Get the selected summary statistics in a torch tensor
    if z_score:
        list_of_statistics = [torch.tensor(zscore(summary_statistics[i])) for i in selected_statistics]
    else:   
        list_of_statistics = [torch.tensor(summary_statistics[i]) for i in selected_statistics]
    selected_summary_statistics = torch.cat(list_of_statistics, dim=0)
    selected_summary_statistics = torch.unsqueeze(selected_summary_statistics, 0)

    # Add theta to the summary statistics if selected
    if theta_selected:
        theta = torch.tensor(summary_statistics["theta"])
        selected_summary_statistics = torch.cat((selected_summary_statistics, theta.T), dim=1)

    # Convert the summary statistics to float32 (required for sbi)
    selected_summary_statistics = selected_summary_statistics.to(torch.float32)
    return selected_summary_statistics



def statistics_from_file(max_files_to_analyze=10):
    folders_inside_statistics = os.listdir("SummaryStatistics")
    folders_inside_statistics.remove("done.txt")
    if ".DS_Store" in folders_inside_statistics:
        folders_inside_statistics.remove(".DS_Store")
    statistics_files = []
    for folder in folders_inside_statistics:
        temp = os.listdir(os.path.join("SummaryStatistics", folder))
        if ".DS_Store" in temp:
            temp.remove(".DS_Store")
        temp = [os.path.join(folder, f) for f in temp]
        statistics_files.extend(temp)
    if len(statistics_files) > max_files_to_analyze:
        statistics_files = statistics_files[:max_files_to_analyze]

    for file in statistics_files:
        with open(os.path.join("SummaryStatistics", file), "rb") as f:
            yield pickle.load(f)

