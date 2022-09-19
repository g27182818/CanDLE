import subprocess
import os
import glob
import numpy as np
from model import *
from datasets import *

# Define the type of sweep to perform
sweep_type = 'mean' # Can be mean, std, random
# Set GPU to train
gpu = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
# Define the number of sweep points
n = 100
# Define the sweep values to train in mean expression, std of expression and random gene subsampling
mean_values = np.linspace(-10.0, 10.0, n)
std_values = np.linspace(-0.01, 6.0, n)
random_frac = np.linspace(1.0, 0.001, n)

# Define experiment names and commands for each of the three possible sweeps

if sweep_type == 'mean':
    # Get experiment names
    exp_names = [os.path.join('mean_thr_exp', f'mean_thr={round(mean_values[i],3)}') for i in range(n)]
    # Get commands to run
    commands = [f'python main.py --mean_thr {mean_values[i]} --std_thr 0.0 --exp_name {exp_names[i]}' for i in range(n)]
elif sweep_type == 'std':
    # Get experiment names
    exp_names = [os.path.join('std_thr_exp', f'std_thr={round(std_values[i],3)}') for i in range(n)]
    # Get commands to run
    commands = [f'python main.py --mean_thr -10.0 --std_thr {std_values[i]} --exp_name {exp_names[i]}' for i in range(n)]

# Make cycle to run all commands serially
for i in range(n):
    print(commands[i])
    command = commands[i].split()
    subprocess.call(command)

