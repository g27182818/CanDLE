import subprocess
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle as pkl
import pandas as pd
import matplotlib.pylab as pylab

# Set figure fontsizes
params = {'legend.fontsize': 'large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)


######################################################################
#            You can safely change these parameters                  #
######################################################################
gpu = '0' # Set GPU to train
mode = 'compute' # Can be compute or plot
######################################################################

# Define cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

if mode == 'compute':
    # Get tissue names using mapper file from toil data
    with open(os.path.join("data", "toil_data", "mappers", "id_2_tissue_mapper.json"), "r") as f:
            mapper_dict = json.load(f)

    # Get all important tissue names sorted 
    tissue_names = sorted(list(set(mapper_dict.values())))
    tissue_names.remove('Not Paired') # Remove not paired tissues
    # Get experiment names and commands to run
    exp_names = [os.path.join('tissues_exp', tissue) for tissue in tissue_names]
    commands = [f'python main.py --exp_name {exp_names[i]} --tissue {tissue_names[i]}' for i in range(len(tissue_names))]

    for command in commands:
        print(command)
        command = command.split()
        subprocess.call(command)