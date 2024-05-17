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

n_files = [75, 100, 120, -1]
selected_stats = [["s_redx"], ["s_redx_cl_lin"], ["s_redx_cl_log"], ["Cxx", "s_redx"], ["Cxx_cl_lin", "s_redx_cl_lin"], ["Cxx", "s_redx", "tucci"], ["Cxx_cl_lin", "s_redx_cl_lin", "tucci"], ["tucci"], ["s_red2", "Cxx"], ["ts_psdx", "tucci"]]
learning_rate = [0.005, 0.0005, 0.00005]
batch_size = [20, 50, 200, 500, 1000]
num_atoms = [5, 10, 20, 50]


def rescale_theta(theta_torch, prior_limits):
    theta_torch2 = theta_torch.clone()
    prior_limits_list = get_prior_limit_list(prior_limits)
    for i in range(theta_torch.shape[1]):
        prior_interval = prior_limits_list[i]
        theta_torch2[:, i] = (theta_torch2[:, i] - np.mean(prior_interval))/(prior_interval[1]-prior_interval[0])
    return theta_torch2


def train_sbi(n_files, selected_stats, learning_rate, batch_size, num_atoms):
    # Get the files
    files = [os.path.join(root, file)
        for root, _, files in os.walk("../../Data/SummaryStatistics/")
        for file in files][:n_files]
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
    prior_limits = {"mu_y": [1e4, 140e4],"k_y": [1.5e-2, 30e-2],"k_int": [1e-3, 6e-3],"tau": [2e-2, 20e-2],"eps": [0.5, 6],}
    prior_box = utils.torchutils.BoxUniform(low=torch.tensor([-0.5]*len(prior_limits)), high=torch.tensor([0.5]*len(prior_limits)))
    best = 0
    for i in range(5):
        prior, num_parameters, prior_returns_numpy = process_prior(prior_box)
        inferece = infer.append_simulations(theta_tot_norm, s_tot)
        density_estimator = infer.train(num_atoms=num_atoms, show_train_summary=False, 
                                    training_batch_size=batch_size, learning_rate=learning_rate) 
        posterior = infer.build_posterior(density_estimator)
        best_i = infer.summary["best_validation_log_prob"]
        if best_i > best:
            best = best_i
            posterior_best = posterior
            inference_best = infer
    
    return (posterior_best, inference_best)


# Make the combinatorial
comb = list(itertools.product(n_files, selected_stats, learning_rate, batch_size, num_atoms))
performance = []
for i in range(len(comb)):
    start = time.time()
    print(f"Combination {i+1}/{len(comb)}: {comb[i]}")
    n_files, selected_stats, learning_rate, batch_size, num_atoms = comb[i]
    results = train_sbi(n_files, selected_stats, learning_rate, batch_size, num_atoms)
    stop = time.time()
    print("Performance: ", results[1]["best_validation_log_prob"], " ; Iteration time: ", stop-start)
    performance.append([comb[i], results])


# Save the results
with open("./CombinatorialTraining.pkl", "wb") as f:
    pickle.dump(performance, f)
