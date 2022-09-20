import subprocess
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd

######################################################################
#            You can safely change these parameters                  #
######################################################################
# Define the type of sweep to perform
sweep_type = 'mean' # Can be mean, std, random
gpu = '1' # Set GPU to train
n = 50 # Number of sweep points
mode = 'plot' # Can be compute or plot
# Define the sweep values to train in mean expression, std of expression and random gene subsampling
mean_values = np.round(np.linspace(-10.0, 10.0, n), 4)
std_values = np.round(np.linspace(-0.01, 6.0, n), 4)
random_frac = np.round(np.linspace(1.0, 0.001, n), 4)
######################################################################

# Handle compute mode
if mode == 'compute':

    # Define cuda device
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

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

# Handle plots (plots are made even with compute method)
if (mode == 'compute') or (mode == 'plot'):
    # Get metric dicts names
    if sweep_type == 'mean':
        metric_paths = sorted(glob.glob(os.path.join('Results', 'mean_thr_exp', '*', 'metric_dicts.pickle')))
        exp_folders = sorted(glob.glob(os.path.join('Results', 'mean_thr_exp', '*')))
        gene_values = pd.read_csv(os.path.join('data', 'toil_data', 'general_stats.csv'), usecols=['joint_mean'])
        x_lab = 'Mean Expression Threshold $[\log_2(TPM+0.001)]$'
        tit = 'Filtering Results for Mean Expression'
        save_path = 'mean_sweep.png'
    elif sweep_type == 'std':
        metric_paths = sorted(glob.glob(os.path.join('Results', 'std_thr_exp', '*', 'metric_dicts.pickle')))
        exp_folders = sorted(glob.glob(os.path.join('Results', 'std_thr_exp', '*')))
        gene_values = pd.read_csv(os.path.join('data', 'toil_data', 'general_stats.csv'), usecols=['joint_std'])
        x_lab = 'Standard Deviation Threshold $[\log_2(TPM+0.001)]$'
        tit = 'Filtering Results for Standard Deviation of Expression'
        save_path = 'std_sweep.png'
    

    # Read all metric paths to a list of metric dicts
    metric_dict_list = [pkl.load(open(metric_path, 'rb')) for metric_path in metric_paths]
    # Obtain val metrics in the last training epoch
    metric_val_dict_list = [metric_dict['val'][-1] for metric_dict in metric_dict_list]

    # Get balanced accuracy for all experiments
    macc_val = [val_metrics['mean_acc'] for val_metrics in metric_val_dict_list]
    # Get thresholds
    threshold_list = [float(folder.split('=')[1]) for folder in exp_folders]
    # Get number of genes in each threshold
    gene_num = [(gene_values.iloc[:,0]>thr).sum() for thr in threshold_list]

    # Construct pandas dataframe with thresholds and metrics
    val_df = pd.DataFrame({'thr': threshold_list, 'macc_val': macc_val, 'gene_num':gene_num}).sort_values('thr')
    general_stats_df = pd.read_csv(os.path.join('data', 'toil_data', 'general_stats.csv'))

    plt.figure()
    general_stats_df.plot(x='joint_mean', y='joint_std', kind='scatter', s=2, alpha=0.4, c='k',
                            xlabel='Mean Expression $[\log_2(TPM+0.001)]$',
                            ylabel='$\sigma$ of Expression $[\log_2(TPM+0.001)]$',
                            title='Standard deviation Vs Mean Expression')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join('Figures', 'gene_scatter_plot.png'), dpi=300)
    plt.close()

    #TODO: Correct Secondary axis
    plt.figure()
    val_df.plot(x='thr', y='macc_val', secondary_y='gene_num', 
                xlabel=x_lab, ylabel='Val Balanced Accuracy', title=tit)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join('Figures', save_path), dpi=300)
    plt.close()

    
