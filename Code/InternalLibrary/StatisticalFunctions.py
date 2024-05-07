import os

import _pickle as pickle
import numpy as np
import torch
from numba import jit
from numpy import (
    array,
    ceil,
    log2,
    max,
    mean,
    median,
    min,
    sum,
    var,
    where,
    zeros,
)
from numpy.fft import fft, fftfreq, ifft
from sbi import utils as utils
from scipy.integrate import cumulative_trapezoid
from scipy.signal import welch


@jit(nopython=True)
def ComputeTheoreticalEntropy(theta):
    """
    Compute the entropy production for the given parameters.

    INPUT
    theta: list of 9 arrays of parameters. The 9 arrays must be all of the same length.

    OUTPUT
    sigmas: entropy production for each simulation
    sigma_mean: mean entropy production
    """

    if len(theta) != 9:
        raise ValueError("There must be 9 parameters in theta")

    if len(set([x.shape for x in theta])) != 1:
        raise Exception("Parameters dimension are not all equal.")

    n_sim = theta[0].shape[0]
    sigmas = np.zeros((n_sim, 1), dtype=np.float64)

    for i in range(n_sim):
        mu_x = theta[0][i]
        mu_y = theta[1][i]
        k_x = theta[2][i]
        k_y = theta[3][i]
        k_int = theta[4][i]
        tau = theta[5][i]
        eps = theta[7][i]

        sigma = (mu_y * eps**2) / ( (1 + k_y * mu_y * tau) - ((k_int**2 * mu_x * mu_y * tau**2) / (1 + k_x * mu_x * tau)))
        sigmas[i] = sigma

    sigma_mean = np.float64(np.mean(sigmas))

    return sigmas, sigma_mean


def ComputeEmpiricalEntropy(x_trace, y_trace, f_trace, theta, n_sim):
    """
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
    """

    Fx = []
    Fy = []
    S_tot = []

    for i in range(n_sim):
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
        F_x = -k_x * x + k_int * y
        F_y = -k_y * y + k_int * x + f
        Fx.append(F_x)
        Fy.append(F_y)

        # Compute the entropy production
        S_x = sum((x_trace[i][1:] - x_trace[i][:-1]) * F_x[:-1] / D_x)
        S_y = sum((f_trace[i][1:] - f_trace[i][:-1]) * F_y[:-1] / D_y)
        S = S_x + S_y
        S_tot.append(S)

    S_mean = mean(S_tot)
    return S_mean, array(Fx), array(Fy), array(S_tot)


def corr(x, y, nmax, dt=False):
    """
    Performs the cross correlation between two single-input signals x and y.

    INPUT
    x: input signal 1
    y: input signal 2
    nmax: maximum number of lags
    dt: time step (default=False)

    OUTPUT
    corr: cross-correlation between x and y
    """

    assert len(x) == len(y), "x and y must have the same length"

    n = len(x)
    # pad 0s to 2n-1
    ext_size = 2 * n - 1
    # nearest power of 2
    fsize = 2 ** ceil(log2(ext_size)).astype("int")

    xp = x - mean(x)
    yp = y - mean(y)

    # do fft and ifft
    cfx = fft(xp, fsize)
    cfy = fft(yp, fsize)
    if dt != False:
        freq = fftfreq(n, d=dt)
        idx = where((freq < -1 / (2 * dt)) + (freq > 1 / (2 * dt)))[0]

        cfx[idx] = 0
        cfy[idx] = 0

    sf = cfx.conjugate() * cfy
    corr = ifft(sf).real
    corr = corr / n

    return corr[:nmax]


def hermite(x, index):
    std_x = np.std(x)
    z = x / std_x
    i = len(index) - 1
    return np.mean(((np.exp(-(z**2) / 2) * np.polynomial.hermite.hermval(x, index) * (2**i * np.math.factorial(i) * np.sqrt(np.pi)) ** -0.5) / np.sqrt(std_x)))


def stat_corr(single_x_trace, single_f_trace, DeltaT, t, t_corr):
    """
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
    """

    sampled_point_amount = single_x_trace.shape[0]
    idx_corr = where((t > 0) * (t < t_corr))[0]
    Cxx = corr(single_x_trace, single_x_trace, sampled_point_amount, dt=DeltaT)  # compute the autocorrellation for each x trace
    Cfx = corr(single_f_trace, single_x_trace, sampled_point_amount, dt=DeltaT)  # compute the cross-correllation for each x and f trace
    Cff = corr(single_f_trace, single_f_trace, sampled_point_amount, dt=DeltaT)  # compute the autocorrellation for each f trace

    return Cxx, Cfx, Cff


def stat_corr_single(single_x_trace, DeltaT, t, t_corr):
    """
    Computes the autocorrelation for a single x trace signal.

    INPUT
    singles_x_trace: single x trace signal
    DeltaT: sampling time
    t: time array
    t_corr: maximum time for the correlation

    OUTPUT
    Cxx: autocorrelation x signal
    """

    sampled_point_amount = single_x_trace.shape[0]
    idx_corr = where((t > 0) * (t < t_corr))[0]
    Cxx = corr(single_x_trace, single_x_trace, sampled_point_amount, dt=DeltaT)  # compute the autocorrellation for each x trace

    return Cxx


def stat_s_redx(Cxx, t_corr, t, theta_i=[1 for i in range(9)]):
    """
    Computes the reduced energy production for a single x trace signal.

    INPUT
    Cxx: autocorrelation signal
    t_corr: maximum time for the correlation
    t: time array
    theta_i: parameters

    OUTPUT
    S_red: reduced x energy production
    """
    mu_x, k_x, D_x = theta_i[0], theta_i[2], theta_i[7]
    S1 = cumulative_trapezoid(Cxx, x=t, axis=-1, initial=0)
    S1 = cumulative_trapezoid(S1, x=t, axis=-1, initial=0)
    idx_corr = where((t > 0) * (t < t_corr))[0]
    S_red1 = Cxx[0] - Cxx[idx_corr]  # First term in S_red
    S_red2 = (((mu_x * k_x) ** 2) * S1[idx_corr] / (D_x * t[idx_corr]))  # Second term in S_red
    S_red = S_red1 + S_red2  # Compute S_red

    return S_red1, S_red2, S_red


def stat_s_redf(Cfx, t_corr, t, theta_i):
    """
    Computes the reduced energy production for a xf trace signal.

    INPUT
    Cxx: autocorrelation signal
    t_corr: maximum time for the correlation
    t: time array
    theta_i: parameters

    OUTPUT
    S_red: reduced f energy production
    """
    mu_x, k_x, D_x = theta_i[0], theta_i[2], theta_i[7]
    idx_corr = where((t > 0) * (t < t_corr))[0]
    S2f = cumulative_trapezoid(Cfx - Cfx[0], x=t, axis=-1, initial=0)
    S3f = cumulative_trapezoid(Cfx, x=t, axis=-1, initial=0)
    S3f = -mu_x * k_x * cumulative_trapezoid(S3f, x=t, axis=-1, initial=0)
    S_redf = 1 - (S2f[idx_corr] + S3f[idx_corr]) / (D_x * t[idx_corr])  # the energy production is to to the fluctuation-dissipation theorem

    return S_redf


def stat_psd(single_trace, nperseg, Sample_frequency):
    """
    Computes the power spectral density for a single trace signal.

    INPUT
    single_trace: single trace signal
    k: number of segments to divide the signal
    Sample_frequency: sampling frequency
    sampled_point_amount: number of sampled points

    OUTPUT
    psd: power spectral density
    """
    frequencies, psd = welch(single_trace, fs=Sample_frequency, nperseg=nperseg)
    return psd


def stat_timeseries(single_timeseries):
    """
    Computes the summary statistics for a single time series signal.

    INPUT
    single_timeseries: single time series signal

    OUTPUT
    s: summary statistics
    """
    statistics_functions = [mean, var, median, max, min]
    s = zeros((len(statistics_functions)))

    for j, func in enumerate(statistics_functions):
        s[j] = func(single_timeseries)

    return s


def stat_hermite(x, index):
    """
    Computes the Hermite statistics for a single trace signal.

    INPUT
    x: single trace signal
    index: Hermite index

    OUTPUT
    s: Hermite statistics
    """
    s = hermite(x, index)
    return s


def compute_summary_statistics(x_trace, theta, DeltaT=1 / 25e3, TotalT=10, n_sim=200):
    summary_statistics = []
    theta = np.array(theta)
    for i in range(n_sim):
        single_summary_statistics = {}
        single_x_trace = x_trace[i, :]
        t = np.linspace(0.0, TotalT, single_x_trace.shape[0])
        t_corr = TotalT / 50

        # Autocorrelation
        Cxx = stat_corr_single(single_x_trace, DeltaT, t, t_corr)
        idx_corr = where((t > 0) * (t < t_corr))[0]
        single_summary_statistics["Cxx"] = Cxx[idx_corr]

        # S red
        S_red1, S_red2, S_red = stat_s_redx(Cxx, t_corr, t)
        single_summary_statistics["s_red1"] = S_red1
        single_summary_statistics["s_red2"] = S_red2
        single_summary_statistics["s_redx"] = S_red

        # Power spectral density
        psdx = stat_psd(single_x_trace, nperseg=1000, Sample_frequency=1 / DeltaT)
        single_summary_statistics["psdx"] = psdx

        # Time series of the power spectral density and the x_trace
        single_summary_statistics["ts_psdx"] = stat_timeseries(psdx)
        single_summary_statistics["ts_x"] = stat_timeseries(single_x_trace)

        # Hermite coefficients
        single_summary_statistics["hermite0"] = stat_hermite(single_x_trace, [1])
        single_summary_statistics["hermite2"] = stat_hermite(single_x_trace, [0, 0, 1])
        single_summary_statistics["hermite4"] = stat_hermite(single_x_trace, [0, 0, 0, 0, 1])
        single_summary_statistics["theta"] = theta[:, i].T

        summary_statistics.append(single_summary_statistics)

    return summary_statistics


def select_summary_statistics(summary_statistics, selected_statistics):
    assert set(selected_statistics).issubset(set(summary_statistics.keys()))
    "The selected statistics are not in the summary statistics"

    # Get the selected summary statistics in a torch tensor
    list_of_statistics = [summary_statistics[i] for i in selected_statistics]
    selected_summary_statistics = torch.tensor(list_of_statistics).float()
    return selected_summary_statistics, torch.tensor(summary_statistics["theta"]).float()


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
