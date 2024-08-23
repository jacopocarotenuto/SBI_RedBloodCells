import sys

sys.path.append("./../..")

from numba import jit
from numba.types import bool_, int_
import numpy as np
import matplotlib.pyplot as plt
from InternalLibrary.StatisticalFunctions import stat_corr_single, stat_s_redx
from scipy.optimize import curve_fit
from scipy.stats import moment

from tqdm import tqdm

from sbi import analysis as analysis
from sbi import utils as utils
from sbi.inference import SNPE, simulate_for_sbi, infer
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
import torch
import statistics

from InternalLibrary.StatisticalFunctions import *
from InternalLibrary.SimulatorPackage import Simulator_noGPU
from pathos.multiprocessing import ProcessingPool as ProcessPool
import pathos
import pickle


# Read the pickle files
with open('posteriors_multiround.pkl', 'rb') as f1:
    pos = pickle.load(f1)

with open('thetas_exp_multiround.pkl', 'rb') as f2:
    thetas_exp = pickle.load(f2)

last = pos[-1]
samples = last.sample((10000,))
fig, ax = analysis.pairplot(samples, figsize=(10, 6), points=thetas_exp.reshape(5,));
plt.show()