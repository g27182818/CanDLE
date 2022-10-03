import subprocess
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd
import matplotlib.pylab as pylab
import torch
import string

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
# Define the type of sweep to perform
sweep_type = 'mean' # Can be mean, std, rand_frac or candle_int
gpu = '0' # Set GPU to train
n = 30 # Number of sweep points
mode = 'plot' # Can be compute or plot
# Define the sweep values to train in mean expression, std of expression and random gene subsampling
mean_values = np.round(np.linspace(-7.5, 12.0, n), 4)
std_values = np.round(np.linspace(0.6, 6.0, n), 4)
rand_frac = np.round(np.logspace(0, -4, n), 4)
candle_int = np.arange(1, 10)
######################################################################

# Define cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def get_sweep_df(path):
    """
    This function receives the directory path of a sweep experiment and returns a pandas dataframe to be plotted

    Args:
        path (str): Sweep experiment directory path. Example: os.path.join('Results', 'mean_thr_sweep')
    Returns:
        sweep_df (pandas.DataFrame): Dataframe containing three columns: thr, macc_val, gene_num
    """
    # breakpoint()
    # Get all metrics paths
    metric_paths = sorted(glob.glob(os.path.join(path, '*', 'metric_dicts.pickle')))
    # Get all experiment folders
    exp_folders = sorted(glob.glob(os.path.join(path, '*')))
    # Get best model paths
    model_paths = sorted(glob.glob(os.path.join(path, '*', 'best_model.pt')))

    # Read all the best models
    models = [torch.load(model_path) for model_path in model_paths]
    # Get best epochs for each experiment
    best_epochs = [m['epoch'] for m in models]
    # Get the number of genes of each model
    gene_num = [m['model_state_dict']['out.weight'].shape[1] for m in models]
    # Read all metric paths to a list of metric dicts
    metric_dict_list = [pkl.load(open(metric_path, 'rb')) for metric_path in metric_paths]
    # Obtain val metrics in the last training epoch
    metric_val_dict_list = [metric_dict_list[i]['val'][best_epochs[i]] for i in range(len(metric_dict_list))]
    # Get balanced accuracy for all experiments
    macc_val = [val_metrics['mean_acc'] for val_metrics in metric_val_dict_list]
    # Get thresholds
    threshold_list = [float(folder.split('=')[1]) for folder in exp_folders]
    
    # Construct pandas dataframe with thresholds and metrics
    sweep_df = pd.DataFrame({'thr': threshold_list, 'macc_val': macc_val, 'gene_num':gene_num}).sort_values('thr')

    return sweep_df


# Handle compute mode
if mode == 'compute':

    # Define experiment names and commands for each of the three possible sweeps
    if sweep_type == 'mean':
        # Get experiment names
        exp_names = [os.path.join('mean_thr_sweep', f'mean_thr={round(mean_values[i],4)}') for i in range(n)]
        # Get commands to run
        commands = [f'python main.py --mean_thr {mean_values[i]} --exp_name {exp_names[i]} --sample_frac 0.5 --epochs -1' for i in range(n)]
    elif sweep_type == 'std':
        # Get experiment names
        exp_names = [os.path.join('std_thr_sweep', f'std_thr={round(std_values[i],4)}') for i in range(n)]
        # Get commands to run
        commands = [f'python main.py --std_thr {std_values[i]} --exp_name {exp_names[i]} --sample_frac 0.5 --epochs -1' for i in range(n)]
    elif sweep_type == 'rand_frac':
        # Get experiment names
        exp_names = [os.path.join('rand_frac_sweep', f'rand_frac={round(rand_frac[i],4)}') for i in range(n)]
        # Get commands to run
        commands = [f'python main.py --rand_frac {rand_frac[i]} --exp_name {exp_names[i]} --sample_frac 0.5 --epochs -1' for i in range(n)]
    elif sweep_type == 'candle_int':
        # Get experiment names
        exp_names = [os.path.join('candle_int_sweep', f'cancer_types={i}') for i in candle_int]
        csv_paths = [os.path.join("Rankings", f'100_candle_thresholds', f'at_least_{i+1}_cancer_types.csv') for i in range(len(exp_names))]
        # Get commands to run
        commands = [f'python main.py --gene_list_csv {csv_paths[i]} --exp_name {exp_names[i]} --sample_frac 0.5 --epochs -1' for i in range(len(exp_names))]

    # Make cycle to run all commands serially
    for i in range(len(commands)):
        print(commands[i])
        command = commands[i].split()
        subprocess.call(command)

# Handle plots
if mode == 'plot':
    
    # Declare labels and titles
    x_lab = ['Mean Expression Threshold $[\log_2(TPM+0.001)]$', 'Standard Deviation Threshold $[\log_2(TPM+0.001)]$',
            'Fraction of Genes']
    tit = ['Sweep Results for Mean Expression', 'Sweep Results for Standard Deviation of Expression',
            'Sweep Results for Random Gene Subsampling']

    # Get general stats dataframe
    general_stats_df = pd.read_csv(os.path.join('data', 'toil_data', 'general_stats.csv'))
    # Get sweep types names
    sweep_types = ['mean_thr_sweep','std_thr_sweep', 'rand_frac_sweep', 'candle_int_sweep']
    sweep_df_list = [] # Declare empty sweep df list

    # Cycle to get sweep dataframe
    for sweep in sweep_types:
        try:
            curr_df = get_sweep_df(os.path.join('Results', sweep))
            sweep_df_list.append(curr_df)
        except:
            msg_str = sweep.split('_')[0]
            print(f'No {sweep} experiments have been made. Please run this code in mode = compute and sweep_type = {msg_str}')

    # Define color for secondary axis
    d_color = 'darkcyan'#'#4c8682'

    
    #################################################################################################################
    #                                    Plot of sweep comparison                                                   #
    #################################################################################################################
    # Plot scatter plot of mean and std for genes
    fig, axes = plt.subplots(figsize=(7.5, 5))
    # Plot every sweep
    sweep_df_list[0].plot(x='gene_num', y='macc_val', ylabel='Validation Balanced Accuracy', ax=axes, legend=None, ylim=(0,1), style='.-', c='k', logx=True)
    sweep_df_list[1].plot(x='gene_num', y='macc_val', ylabel='Validation Balanced Accuracy', ax=axes, legend=None, ylim=(0,1), style='.-', c='darkcyan', logx=True)
    sweep_df_list[2].plot(x='gene_num', y='macc_val', ylabel='Validation Balanced Accuracy', ax=axes, legend=None, ylim=(0,1), style='.-', c='dodgerblue', logx=True)
    sweep_df_list[3].plot(x='gene_num', y='macc_val', ylabel='Validation Balanced Accuracy', ax=axes, legend=None, ylim=(0,1), style='.-', c='red', logx=True,
                          title='Comparison of Gene Subsampling Methods', xlabel='Number of Genes', xlim=(0,None))
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.legend(['Mean Thresholding', 'Std. Thresholding', 'Random Subsampling', 'CanDLE Interpretation'], loc='center right')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join('Figures', 'sweep_comparison.png'), dpi=300)
    plt.close()


breakpoint()
