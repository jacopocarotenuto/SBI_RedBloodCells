import os
import sys
sys.path.append("./../..")

from InternalLibrary.StatisticalFunctions import *
from InternalLibrary.SimulatorPackage import Simulator_noGPU

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import torch

from tqdm import tqdm
import _pickle as pickle

import sympy as sym

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import SNPE, SNPE_A, SNPE_C, simulate_for_sbi, infer
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)


dt = 1e-6# The time step size for the simulation. This is the amount of time that passes in the model for each step of the simulation
Sample_frequency = 25_000 # The sampling frequency. This is the number of samples that are recorded per unit time
DeltaT = 1/Sample_frequency # The sampling period. This is the amount of time that passes in the model between each sample that is recorded
TotalT = 13 # The total time for the simulation. This is the total amount of time that the simulation is intended to represent
transient = 3
t_corr = TotalT/50

time_steps_amount = np.int64((TotalT-transient)/dt) # Number of steps
sampled_point_amount = np.int64((TotalT-transient)/DeltaT) # Number of sampled points
sampling_delta_time_steps = np.int64(DeltaT/dt) # Number of steps between samples
t = np.linspace(0., TotalT - transient, sampled_point_amount) # Time array

prior_limits = {
    "mu_y": [1e4, 140e4],
    "k_y": [1.5e-2, 30e-2],
    "k_int": [1e-3, 6e-3],
    "tau": [2e-2, 20e-2],
    "eps": [0.5, 6],
}


def helperLoad(selected_stats):
    z_score = False # z-score normalization of the summary statistics
    cl_log = 5 # post-subsampling of Cxx and s_redx (-1 for no post-subsampling)
    cl_lin = -1
    fit_corr = False
    fit_s_redx = False # Implemented: "exp" or "simple"
    skiped_simulations = 0

    # List file in the directory SummaryStatistics
    #files = os.listdir("../../Data/SummaryStatistics/20240515/")[:-1]
    files = [os.path.join(root, file)
            for root, _, files in os.walk("../../Data/SummaryStatistics/")
            for file in files][1:-1]
    print("Reading ", len(files), " file for a total of ", 200*len(files), " simulations... \n")
    loaded_simul = 0

    # Pipeline from the n_sim simulations to the be ready for training
    first = True
    for file in files:
        print("Reading file", file, " i =", loaded_simul, " / ", 200*len(files), end='\r')
        with open(f"{file}", "rb") as f:
            summary_stats_batch = pickle.load(f)
        
        for i in range(len(summary_stats_batch)):
            # Here handle the local simulations
            summary_stats = summary_stats_batch[i]
            s_i = select_summary_statistics(summary_stats, selected_stats, DeltaT,
                    z_score=z_score, cl_lin=cl_lin, cl_log=cl_log, fit_cxx=fit_corr, fit_s_redx=fit_s_redx)
            
            if s_i == None: 
                skiped_simulations += 1
                print("Skiped simulations: ", skiped_simulations, end='\r')
                continue
            
            if first:
                s_tot = s_i
                theta_tot = torch.from_numpy(summary_stats["theta"]).to(torch.float32)
                summary_stats_length = [len(summary_stats[s]) for s in selected_stats]
                print("Length of each summary statistics: ", summary_stats_length)
                first = False
            else:
                s_tot = torch.cat((s_tot, s_i), dim=0)
                theta_tot = torch.cat((theta_tot, torch.from_numpy(summary_stats["theta"]).to(torch.float32)), dim=1)
            
            loaded_simul += 1

    theta_tot = theta_tot.T

    return theta_tot, s_tot


def helperComparison(theta_tot_norm, s_tot, n_trials):
    nn_performance = np.zeros((n_trials))
    mse_mean = np.zeros((n_trials))
    mse_mode = np.zeros((n_trials))

    print("Starting comparison")
    for i in tqdm(range(n_trials)):
        infer = SNPE(prior=prior)
        inferece = infer.append_simulations(theta_tot_norm, s_tot)

        density_estimator = infer.train(num_atoms=20, show_train_summary=False, 
                                        training_batch_size=500, learning_rate=0.005,
                                        stop_after_epochs=50)
    
        nn_performance[i] = infer.summary["best_validation_log_prob"][0]

        posterior = infer.build_posterior(density_estimator)
        sigma_true, sigma_posterior = CompareTheoreticalSigma(posterior, 10, 1e5, return_theta=False,
                  selected_stats=selected_stats, cl_log=5)
        # Check for negative values (failed fit)
        failed_trials = np.all(sigma_posterior < 0, axis=1)
        if np.sum(failed_trials) > 0: print(np.sum(failed_trials))
        sigma_true = sigma_true[~failed_trials]
        sigma_posterior = sigma_posterior[~failed_trials]
        # Do the actual comparison
        mean_array = np.mean(sigma_posterior, axis=1)
        mode_array = np.array([get_mode(sigma_posterior[i]) for i in range(len(sigma_true))])
        mse_mean[i] = np.mean((sigma_true.reshape(sigma_true.shape[0])-mean_array)**2)
        mse_mode[i] = np.mean((sigma_true.reshape(sigma_true.shape[0])-mode_array)**2)

    return nn_performance, mse_mean, mse_mode


results = {}
selected_stats_set = [["Cxx"], ["Cxx", "s_redx"]]
sample_sizes = [5_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000]

results["stats"] = selected_stats_set
results["sizes"] = sample_sizes

print("Stating the routine")
for i in range(len(selected_stats_set)):
    selected_stats = selected_stats_set[i]
    print("\n Starting ", selected_stats)

    # Load the full dataset
    theta_tot, s_tot = helperLoad(selected_stats)
    theta_tot_norm = rescale_theta(theta_tot, prior_limits)
    max_idx = theta_tot.shape[0]

    # Initialize the arrays for the results
    nn_perf_array = np.zeros((len(sample_sizes)))
    mse_mean_array = np.zeros((len(sample_sizes)))
    mse_mode_array = np.zeros((len(sample_sizes)))

    for s in range(len(sample_sizes)): 
        size = sample_sizes[s]
        # Sample the dataset
        indeces = torch.randint(0, max_idx, (size,))
        theta_tot_norm_sub = theta_tot_norm[indeces]
        s_tot_sub = s_tot[indeces]

        # SBI routine
        prior_box = utils.torchutils.BoxUniform(low=torch.tensor([-0.5]*len(prior_limits)), high=torch.tensor([0.5]*len(prior_limits)))
        prior, num_parameters, prior_returns_numpy = process_prior(prior_box)

        nn_performance, mse_mean, mse_mode = helperComparison(theta_tot_norm_sub, s_tot_sub, n_trials=1)

        # Save results into the array
        nn_perf_array[s] = nn_performance[0]
        mse_mean_array[s] = mse_mean[0]
        mse_mode_array[s] = mse_mode[0]

    # Save the results
    results[i] = [nn_perf_array, mse_mean_array, mse_mode_array]


with open("comparisonSampleSize.pkl", 'wb') as f:
    pickle.dump(results, f)


    

