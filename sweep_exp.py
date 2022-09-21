import subprocess
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
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
# Define the type of sweep to perform
sweep_type = 'candle_int' # Can be mean, std, rand_frac or candle_int
gpu = '0' # Set GPU to train
n = 50 # Number of sweep points
mode = 'compute' # Can be compute or plot
# Define the sweep values to train in mean expression, std of expression and random gene subsampling
mean_values = np.round(np.linspace(-10.0, 10.0, n), 4)
std_values = np.round(np.linspace(-0.01, 6.0, n), 4)
rand_frac = np.round(np.linspace(1.0, 0.001, n), 4)
candle_int = np.arange(1, 15)
######################################################################

# Define cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = gpu


def sweep_exp_path_2_general_key(path):
    """
    This function takes a sweep experiment path and returns the key the parameter values in 
    the joint general stats dataframe.

    Args:
        path (str): Sweep experiment path. Example: os.path.join('Results', 'mean_thr_sweep')

    Returns:
        str: key to search in general stats dataframe. Example: 'joint_mean'
    """
    # Get name of the last folder in path
    folder_name = os.path.split(path)[1]
    # Get parameter name
    param_name = folder_name.split('_')[0]
    # Get key name for general stats
    key_name = f'joint_{param_name}'
    return key_name



def get_sweep_df(path):
    """
    This function receives the directory path of a sweep experiment and returns a pandas dataframe to be plotted

    Args:
        path (str): Sweep experiment directory path. Example: os.path.join('Results', 'mean_thr_sweep')
    Returns:
        sweep_df (pandas.DataFrame): Dataframe containing three columns: thr, macc_val, gene_num
    """

    # Get all metrics paths
    metric_paths = sorted(glob.glob(os.path.join(path, '*', 'metric_dicts.pickle')))
    # Get all experiment folders
    exp_folders = sorted(glob.glob(os.path.join(path, '*')))
    # Get general stats for all genes
    general_stats = pd.read_csv(os.path.join('data', 'toil_data', 'general_stats.csv'))


    # Read all metric paths to a list of metric dicts
    metric_dict_list = [pkl.load(open(metric_path, 'rb')) for metric_path in metric_paths]
    # Obtain val metrics in the last training epoch
    metric_val_dict_list = [metric_dict['val'][-1] for metric_dict in metric_dict_list]
    # Get balanced accuracy for all experiments
    macc_val = [val_metrics['mean_acc'] for val_metrics in metric_val_dict_list]
    # Get thresholds
    threshold_list = [float(folder.split('=')[1]) for folder in exp_folders]

    # Get key in general stats dataframe
    search_key = sweep_exp_path_2_general_key(path)

    # Handle the possible random subsampling of the genes
    if search_key not in general_stats.columns:
        gene_num = [np.round(general_stats.shape[0]*frac) for frac in threshold_list] # This handles the random subsampling
    else:
        # Get number of genes in each threshold
        gene_num = [(general_stats[search_key]>thr).sum() for thr in threshold_list]

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
        commands = [f'python main.py --mean_thr {mean_values[i]} --std_thr -0.01 --rand_frac 1.0 --exp_name {exp_names[i]}' for i in range(n)]
    elif sweep_type == 'std':
        # Get experiment names
        exp_names = [os.path.join('std_thr_sweep', f'std_thr={round(std_values[i],4)}') for i in range(n)]
        # Get commands to run
        commands = [f'python main.py --mean_thr -10.0 --std_thr {std_values[i]} --rand_frac 1.0 --exp_name {exp_names[i]}' for i in range(n)]
    elif sweep_type == 'rand_frac':
        # Get experiment names
        exp_names = [os.path.join('rand_frac_sweep', f'rand_frac={round(rand_frac[i],4)}') for i in range(n)]
        # Get commands to run
        commands = [f'python main.py --mean_thr -10.0 --std_thr -0.01 --rand_frac {rand_frac[i]} --exp_name {exp_names[i]}' for i in range(n)]
    elif sweep_type == 'candle_int':
        # Get experiment names
        exp_names = [os.path.join('candle_int_sweep', f'cancer_types={i}') for i in candle_int]
        csv_paths = [os.path.join("Rankings", f'100_candle_thresholds', f'at_least_{i+1}_cancer_types.csv') for i in range(len(exp_names))]
        # Get commands to run
        commands = [f'python main.py --mean_thr -10.0 --std_thr -0.01 --rand_frac 1.0 --gene_list_csv {csv_paths[i]} --exp_name {exp_names[i]}' for i in range(len(exp_names))]

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
    sweep_types = ['mean_thr_sweep','std_thr_sweep', 'rand_frac_sweep']
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
    #                                    Plot of individual sweeps                                                  #
    #################################################################################################################
    # Plot scatter plot of mean and std for genes
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
    general_stats_df.plot(x='joint_mean', y='joint_std', kind='scatter', s=2, alpha=0.4, c='k',
                            xlabel='Mean Expression $[\log_2(TPM+0.001)]$',
                            ylabel='$\sigma$ of Expression $[\log_2(TPM+0.001)]$',
                            title='Standard deviation Vs Mean Expression', ax=axes[0,0],
                            xlim=(general_stats_df['joint_mean'].min(),general_stats_df['joint_mean'].max()),
                            ylim=(general_stats_df['joint_std'].min(),general_stats_df['joint_std'].max()))
    axes[0,0].spines['top'].set_visible(False)
    axes[0,0].spines['right'].set_visible(False)

    # Plot results of mean sweep
    sweep_df_list[0].plot(x='thr', y='macc_val', xlabel=x_lab[0], ylabel='Val Balanced Accuracy', title=tit[0],
                    ylim=(0,1), xlim=(general_stats_df['joint_mean'].min(),general_stats_df['joint_mean'].max()),
                    style='.-', c='k', legend=None, ax=axes[1,0])
    
    ax1 = plt.twinx(axes[1,0])
    ax1.spines['top'].set_visible(False)
    axes[1,0].spines['top'].set_visible(False)
    sweep_df_list[0].plot(x='thr', y='gene_num', ylabel='Number of genes', ax=ax1, legend=None, ylim=(0,None), style='.-', c=d_color)
    ax1.spines['right'].set_color(d_color)
    ax1.tick_params(axis='y', colors=d_color)
    ax1.yaxis.label.set_color(d_color)

    # Plot results of std sweep
    sweep_df_list[1].plot(x='thr', y='macc_val', xlabel=x_lab[1], ylabel='Val Balanced Accuracy', title=tit[1],
                    ylim=(0,1), xlim=(general_stats_df['joint_std'].min(),general_stats_df['joint_std'].max()),
                    style='.-', c='k', legend=None, ax=axes[0,1])
    
    ax1 = plt.twinx(axes[0,1])
    ax1.spines['top'].set_visible(False)
    axes[0,1].spines['top'].set_visible(False)
    sweep_df_list[1].plot(x='thr', y='gene_num', ylabel='Number of genes', ax=ax1, legend=None, ylim=(0,None), style='.-', c=d_color)
    ax1.spines['right'].set_color(d_color)
    ax1.tick_params(axis='y', colors=d_color)
    ax1.yaxis.label.set_color(d_color)

    # Plot results of rand_frac sweep
    sweep_df_list[2].plot(x='thr', y='macc_val', xlabel=x_lab[2], ylabel='Val Balanced Accuracy', title=tit[2],
                    ylim=(0,1), xlim=(0,1),
                    style='.-', c='k', legend=None, ax=axes[1,1])
    
    ax1 = plt.twinx(axes[1,1])
    ax1.spines['top'].set_visible(False)
    axes[1,1].spines['top'].set_visible(False)
    sweep_df_list[2].plot(x='thr', y='gene_num', ylabel='Number of genes', ax=ax1, legend=None, ylim=(0,None), style='.-', c=d_color)
    ax1.spines['right'].set_color(d_color)
    ax1.tick_params(axis='y', colors=d_color)
    ax1.yaxis.label.set_color(d_color)

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join('Figures', 'thresholding.png'), dpi=300)
    plt.close()

    #################################################################################################################
    #                                    Plot of sweep comparison                                                   #
    #################################################################################################################
    # Plot scatter plot of mean and std for genes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    general_stats_df.plot(x='joint_mean', y='joint_std', kind='scatter', s=2, alpha=0.4, c='k',
                            xlabel='Mean Expression $[\log_2(TPM+0.001)]$',
                            ylabel='$\sigma$ of Expression $[\log_2(TPM+0.001)]$',
                            title='Standard deviation Vs Mean Expression', ax=axes[0],
                            xlim=(general_stats_df['joint_mean'].min(),general_stats_df['joint_mean'].max()),
                            ylim=(general_stats_df['joint_std'].min(),general_stats_df['joint_std'].max()))
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Plot every sweep
    sweep_df_list[0].plot(x='gene_num', y='macc_val', ylabel='Val Balanced Accuracy', ax=axes[1], legend=None, ylim=(0,1), style='.-', c='k')
    sweep_df_list[1].plot(x='gene_num', y='macc_val', ylabel='Val Balanced Accuracy', ax=axes[1], legend=None, ylim=(0,1), style='.-', c='darkcyan')
    sweep_df_list[2].plot(x='gene_num', y='macc_val', ylabel='Val Balanced Accuracy', ax=axes[1], legend=None, ylim=(0,1), style='.-', c='dodgerblue',
                          title='Comparison of Gene Subsampling Methods', xlabel='Number of Genes', xlim=(0,None))
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    plt.legend(['Mean Thresholding', 'Std. Thresholding', 'Random Subsampling'], loc='center right')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join('Figures', 'sweep_comparison.png'), dpi=300)
    plt.close()
