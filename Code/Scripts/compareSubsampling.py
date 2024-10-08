import sys
sys.path.append("./../..")
from InternalLibrary.StatisticalFunctions import *
import numpy as np
import os
from sbi.utils.user_input_checks import process_prior
from sbi.inference import SNPE, infer
import itertools
import _pickle as pickle
import torch
from sbi import utils as utils
import time as time
from skopt import gp_minimize
from skopt.space import Categorical
from pathos.multiprocessing import ProcessPool
import pathos.multiprocessing as multiprocessing

# Helper function
def rescale_theta(theta_torch, prior_limits):
    theta_torch2 = theta_torch.clone()
    prior_limits_list = get_prior_limit_list(prior_limits)
    for i in range(theta_torch.shape[1]):
        prior_interval = prior_limits_list[i]
        theta_torch2[:, i] = (theta_torch2[:, i] - np.mean(prior_interval))/(prior_interval[1]-prior_interval[0])
    return theta_torch2


"""
# This first part of the code is meant to perform a Bayesian optimization over a large set of (hyper)parameters

# Define the space for the optimization
n_files = [75, 100, 120, -1]
selected_stats = [
    ("s_redx",), ("s_redx_cl_lin",), ("s_redx_cl_log",), 
    ("Cxx", "s_redx",), ("Cxx_cl_lin", "s_redx_cl_lin",), 
    ("Cxx", "s_redx", "tucci",), ("Cxx_cl_lin", "s_redx_cl_lin", "tucci",), 
    ("tucci",), ("s_red2", "Cxx",), ("ts_psdx", "tucci",)]
learning_rate = [0.005, 0.0005, 0.00005]
batch_size = [20, 50, 200, 500, 1000]
num_atoms = [5, 10, 20, 50]

space = [
    Categorical(n_files, name='n_files'),
    Categorical(selected_stats, name='selected_stats'),
    Categorical(learning_rate, name='learning_rate'),
    Categorical(batch_size, name='batch_size'),
    Categorical(num_atoms, name='num_atoms')
]

# Loss function
def train_sbi(params):
    n_files, selected_stats, learning_rate, batch_size, num_atoms = params
    print("Trying: ", n_files, selected_stats, learning_rate, batch_size, num_atoms)
    prior_limits = {"mu_y": [1e4, 140e4],"k_y": [1.5e-2, 30e-2],"k_int": [1e-3, 6e-3],"tau": [2e-2, 20e-2],"eps": [0.5, 6],}
    
    # Get the files
    files = [os.path.join(root, file)
        for root, _, files in os.walk("../../Data/SummaryStatistics/")
        for file in files][1:n_files]
    z_score = True
    
    # Compute the tensors for the training
    first = True
    for file in files:
        with open(f"{file}", "rb") as f:
            summary_stats_batch = pickle.load(f)
        for i in range(len(summary_stats_batch)):
            summary_stats = summary_stats_batch[i]
            s_i = select_summary_statistics(summary_stats, selected_stats, z_score=z_score)
            if first:
                s_tot = s_i
                theta_tot = torch.from_numpy(summary_stats["theta"]).to(torch.float32)
                first = False
            else:
                s_tot = torch.cat((s_tot, s_i), dim=0)
                theta_tot = torch.cat((theta_tot, torch.from_numpy(summary_stats["theta"]).to(torch.float32)), dim=1)        
    theta_tot = theta_tot.T
    theta_tot_norm = rescale_theta(theta_tot, prior_limits)

    # Train the model and evaluate performance
    prior_box = utils.torchutils.BoxUniform(low=torch.tensor([-0.5]*len(prior_limits)), high=torch.tensor([0.5]*len(prior_limits)))
    best = 0
    for i in range(1):
        prior, num_parameters, prior_returns_numpy = process_prior(prior_box)
        infer = SNPE(prior=prior)
        inferece = infer.append_simulations(theta_tot_norm, s_tot)
        density_estimator = infer.train(num_atoms=num_atoms, show_train_summary=False, 
                                    training_batch_size=batch_size, learning_rate=learning_rate) 
        posterior = infer.build_posterior(density_estimator)
        best_i = infer.summary["best_validation_log_prob"][0]
        if best_i > best:
            best = best_i

    # Write on file    
    with open("./CombinatorialTraining.txt", "a") as f:
        f.write(f"Parameters: {params}\n")
        f.write(f"Loss: {-best}\n")

    # Return the loss function
    return -best


# Make the combinatorial (the hard way)
# comb = list(itertools.product(n_files, selected_stats, learning_rate, batch_size, num_atoms))
# performance = []
# for i in range(len(comb)):
#     start = time.time()
#     print(f"Combination {i+1}/{len(comb)}: {comb[i]}")
#     n_files, selected_stats, learning_rate, batch_size, num_atoms = comb[i]
#     results = train_sbi(n_files, selected_stats, learning_rate, batch_size, num_atoms)
#     stop = time.time()
#     print("Loss: ", result, " ; Iteration time: ", stop-start)
#     performance.append([comb[i], results])


# Optimize the hyperparameters
results = gp_minimize(train_sbi, space, n_calls=30, random_state=0, n_jobs=-1, verbose=True)


# Save the results
with open("./CombinatorialTraining.pkl", "wb") as f:
    pickle.dump(results, f)
"""


# In this second part, we perform a scan over some parameters
cl_sizes = [3, 5, 10, 20, 30, 50, 100, 200, 500]
selected_stats = [["Cxx"], ["Cxx", "s_redx"], ["s_redx"], ["Cxx", "s_red2"], ["Cxx", "s_redx", "tucci"]]


def train_sbi(selected_stats, cl_lin, cl_log):
    prior_limits = {"mu_y": [1e4, 140e4],"k_y": [1.5e-2, 30e-2],"k_int": [1e-3, 6e-3],"tau": [2e-2, 20e-2],"eps": [0.5, 6],}
    num_atoms = 20; learning_rate = 0.005; batch_size=500

    # Get the files
    files = [os.path.join(root, file)
        for root, _, files in os.walk("../../Data/SummaryStatistics/")
        for file in files][1:] # Loading all files (1: to avoid done.txt)
    z_score = True
    
    # Compute the tensors for the training
    first = True
    for file in files:
        with open(f"{file}", "rb") as f:
            summary_stats_batch = pickle.load(f)
        for i in range(len(summary_stats_batch)):
            summary_stats = summary_stats_batch[i]
            s_i = select_summary_statistics(summary_stats, selected_stats, z_score=z_score, cl_lin=cl_lin, cl_log=cl_log)
            if first:
                s_tot = s_i
                theta_tot = torch.from_numpy(summary_stats["theta"]).to(torch.float32)
                first = False
            else:
                s_tot = torch.cat((s_tot, s_i), dim=0)
                theta_tot = torch.cat((theta_tot, torch.from_numpy(summary_stats["theta"]).to(torch.float32)), dim=1)        
    theta_tot = theta_tot.T
    theta_tot_norm = rescale_theta(theta_tot, prior_limits)

    # Train the model and evaluate performance (multiple times for statistical purpouses)
    prior_box = utils.torchutils.BoxUniform(low=torch.tensor([-0.5]*len(prior_limits)), high=torch.tensor([0.5]*len(prior_limits)))
    best = 0
    for i in range(2):
        prior, num_parameters, prior_returns_numpy = process_prior(prior_box)
        infer = SNPE(prior=prior)
        inferece = infer.append_simulations(theta_tot_norm, s_tot)
        density_estimator = infer.train(num_atoms=num_atoms, show_train_summary=False, 
                                    training_batch_size=batch_size, learning_rate=learning_rate) 
        posterior = infer.build_posterior(density_estimator)
        best_i = infer.summary["best_validation_log_prob"][0]
        if best_i > best:
            best = best_i
    
    print("Using ", selected_stats, "; cl_lin = ", cl_lin, "; cl_log = ", cl_log, "; Loss function: ", -best)
    # Return the loss function
    return -best


# Two for loops approach
cl_lin_results = np.zeros((len(selected_stats), len(cl_sizes)))
cl_log_results = np.zeros((len(selected_stats), len(cl_sizes)))
    
for s, stats in enumerate(selected_stats):
    for c, size in enumerate(cl_sizes):
        a = time.time()
        # Linear subsample
        res = train_sbi(stats, cl_lin=size, cl_log=-1)
        cl_lin_results[s, c] = res
        # Log subsample
        res = train_sbi(stats, cl_lin=-1, cl_log=size)
        cl_log_results[s, c] = res
        b = time.time()
        print("Time taken: ", b-a)


# # Define your function to run in parallel
# def parallel_task(params):
#     s, stats, c, size = params
#     lin_result = train_sbi(stats, cl_lin=size, cl_log=-1)
#     log_result = train_sbi(stats, cl_lin=-1, cl_log=size)
#     return s, c, lin_result, log_result
# # Initialize your variables
# cl_lin_results = np.zeros((len(selected_stats), len(cl_sizes)))
# cl_log_results = np.zeros((len(selected_stats), len(cl_sizes)))
# # Prepare the parameters for parallel execution
# params = [(s, stats, c, size) for s, stats in enumerate(selected_stats) for c, size in enumerate(cl_sizes)]
# # Use ProcessPool to execute the tasks in parallel
# cores = multiprocessing.cpu_count()
# with ProcessPool(nodes=cores) as pool:
#     results = pool.map(parallel_task, params)
# # Store the results
# for s, c, lin_result, log_result in results:
#     cl_lin_results[s, c] = lin_result
#     cl_log_results[s, c] = log_result


# Save the results in pickles
with open("./cl_lin_results.pkl", "wb") as f:
    pickle.dump(cl_lin_results, f)
with open("./cl_log_results.pkl", "wb") as f:
    pickle.dump(cl_log_results, f)