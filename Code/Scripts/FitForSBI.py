import sys
sys.path.append("./../..")

from InternalLibrary.StatisticalFunctions import *
from InternalLibrary.SimulatorPackage import Simulator_noGPU
import numpy as np
import torch
from tqdm import tqdm
import _pickle as pickle
from scipy.optimize import curve_fit

## Parameters
dt = 1e-6
Sample_frequency = 25_000 
DeltaT = 1/Sample_frequency  
TotalT = 13
transient = 3
EffectiveT = TotalT - transient
sampled_point_amount = np.int64((EffectiveT)/DeltaT) 
t = np.linspace(0., EffectiveT, sampled_point_amount) 


## Functions
def stat_fit_corr(single_corr, DeltaT):
    """
    Fit the correlation function with a sum of 3 exponentials
    """
    t_cxx = np.linspace(0., (len(single_corr)+1)*DeltaT, (len(single_corr)+1))[1:]
    
    def cxx_exp3(t, a1, a2, a3, b1, b2, b3):
        return a1*np.exp(-b1*t) + a2*np.exp(-b2*t) + a3*np.exp(-b3*t)
    
    try: 
        popt, pcov = curve_fit(cxx_exp3, t_cxx, single_corr, p0=[1e2, 1e2, 1e2, 10, 10, 10], maxfev=5000)
    except:
        return np.zeros(6)

    return popt, cxx_exp3(t_cxx, *popt)


def stat_fit_s_redx(single_s_redx, DeltaT, mode="exp"):
    """
    Fit the s_redx function
    """
    assert mode in ["exp", "simple"], "Mode not recognized"

    t_cxx = np.linspace(0., (len(single_s_redx)+1)*DeltaT, (len(single_s_redx)+1))[1:]
    
    def s_redx_simple(t, a, tau):
        return(1 + a*t/(1+t/tau))

    def s_redx_exp(t, a1, a2, b1, b2, b3, c):
        a3 = 1 - a1 - a2 
        sum_exp = a1*np.exp(-b1*t) + (a2)*np.exp(-(b1+b2)*t) + (a3)*np.exp(-(b1+b2+b3)*t)
        sum = a1*b1 + (a2)*(b1+b2) + (a3)*(b1+b2+b3)
        tau = 1/sum
        return(1 + c - (c*tau/t)*(1-sum_exp))
    
    if mode == "exp":
        try: 
            popt, pcov = curve_fit(s_redx_exp, t_cxx, single_s_redx, p0=[1, 10, 10, 0.1, 0.01, 10],
                          bounds=([0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]), maxfev=5000)
        except:
            return np.zeros(6)
        return popt, s_redx_exp(t_cxx, *popt)

    if mode == "simple":
        try:
            popt, pcov = curve_fit(s_redx_simple, t_cxx, single_s_redx, p0=[1e3, 1e-2],
                          bounds=([0, 0], [np.inf, np.inf]), maxfev=5000)
        except:
            return np.zeros(6)
        return popt, s_redx_simple(t_cxx, *popt)
        

def select_summary_statistics(summary_statistics, selected_statistics, DeltaT,
                              z_score=False, cl_lin=-1, cl_log=-1, fit_cxx=False, fit_s_redx=False):
    selected_statistics = selected_statistics.copy()
    
    # Check that the selected statistics are in the summary statistics
    assert set(selected_statistics).issubset(set(summary_statistics.keys()))
    "The selected statistics are not in the summary statistics"

    # Checks on postprocessing
    assert cl_log < 0 or cl_lin < 0, "You cannot subsample bot 'lin' and 'log' at the same time"
    if cl_lin > 0 or cl_log > 0: assert (fit_cxx == False) and (fit_s_redx == False), "You cannot subsample and fit at the same time"
    
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

    # Fit Cxx and s_redx and use the paramaters as summary statistics
    if fit_cxx and "Cxx" in selected_statistics:
        summary_statistics["Cxx"] = stat_fit_corr(summary_statistics["Cxx"], DeltaT)[0]
        if (summary_statistics["Cxx"] == np.zeros(6)).all(): return None

    if fit_s_redx and "s_redx" in selected_statistics:
        summary_statistics["s_redx"] = stat_fit_s_redx(summary_statistics["s_redx"], DeltaT, mode=fit_s_redx)[0]
        if (summary_statistics["s_redx"] == np.zeros(6)).all(): return None
        
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


def do(selected_stats):
    z_score = False # z-score normalization of the summary statistics
    cl_log = -1 # post-subsampling of Cxx and s_redx (-1 for no post-subsampling)
    cl_lin = -1
    fit_corr = True 
    fit_s_redx = "exp" # Implemented: "exp" or "simple"
    skiped_simulations = 0

    print("Doing: ", selected_stats)

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
                continue
            
            if first:
                s_tot = s_i
                theta_tot = torch.from_numpy(summary_stats["theta"]).to(torch.float32)
                summary_stats_length = [len(summary_stats[s]) for s in selected_stats]
                #print("Length of each summary statistics: ", summary_stats_length)
                first = False
            else:
                s_tot = torch.cat((s_tot, s_i), dim=0)
                theta_tot = torch.cat((theta_tot, torch.from_numpy(summary_stats["theta"]).to(torch.float32)), dim=1)
            
            loaded_simul += 1

    print("Done, failed simulations", skiped_simulations)
    theta_tot = theta_tot.T
    return s_tot, theta_tot
    

# Do the selection of the summary statistics
selected_stats = ["s_redx"]
s_tot, theta_tot = do(selected_stats)
with open("./data/selected_s_redx.pkl", "wb") as f:
    pickle.dump({"s_tot": s_tot, "theta_tot": theta_tot}, f)

selected_stats = ["s_redx", "Cxx"]
s_tot, theta_tot = do(selected_stats)
with open("./data/selected_s_redx_Cxx.pkl", "wb") as f:
    pickle.dump({"s_tot": s_tot, "theta_tot": theta_tot}, f)

selected_stats = ["Cxx"]
s_tot, theta_tot = do(selected_stats)
with open("./data/selected_Cxx.pkl", "wb") as f:
    pickle.dump({"s_tot": s_tot, "theta_tot": theta_tot}, f)

selected_stats = ["Cxx", "tucci"]
s_tot, theta_tot = do(selected_stats)
with open("./data/selected_Cxx_tucci.pkl", "wb") as f:
    pickle.dump({"s_tot": s_tot, "theta_tot": theta_tot}, f)