import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle as pkl
import matplotlib
from adjustText import adjust_text
import string
import glob
from matplotlib.colors import LinearSegmentedColormap
from datasets import *
from model import *
from utils import *

# Get Parser
parser = get_dataset_parser()
parser.add_argument('--weights',        type=str,       default="True",     help="Whether to use weights in model", choices=["True", "False"])
parser.add_argument('--mode',           type=str,       default="compute",  help="Mode to run the code.",           choices=["compute", "plot"])
parser.add_argument('--gpu',            type=str,       default="0",        help="GPU on which to run experiments")
# Parse the argument
args = parser.parse_args()
args_dict = vars(args)


# Set GPU in operating system
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Define experiment name
exp_folder_name = f'CanDLE_all_vs_one_{args.source}_weights_{args.weights}_sample_frac_{args.sample_frac}'

# Get mapper file to know labels depending on the source
if (args.source == 'toil') or (args.source == 'recount3') :
    mapper_path = os.path.join('data','toil_data', 'mappers', 'category_mapper.json')
elif args.source == 'wang':
    mapper_path = os.path.join('data','wang_data', 'mappers', 'wang_standard_label_mapper.json')
else:
    pass

# Load the mapper file
with open(mapper_path, "r") as f:
    category_mapper = json.load(f)

# Get a list of the available labels
labels = list(category_mapper.values())

# Filter the labels depending on the dataset
if args.dataset == 'both':
    pass
elif args.dataset == 'tcga':
    # Delete all label that start with 'GTEX'
    labels = [label for label in labels if not label.startswith('GTEX')]
elif args.dataset == 'gtex':
    # Delete all label that start with 'TCGA'
    labels = [label for label in labels if not label.startswith('TCGA')]
else:
    raise ValueError('dataset must be either "tcga", "gtex" or "both"')

# # To test code. comment in real use
# labels = labels[:2]

# Declare folder names for all experiments
exp_names = [os.path.join(exp_folder_name, args.dataset, label) for label in labels]


# If mode is 'compute', compute all experiments
if args.mode == 'compute':
    for i in range(len(labels)):
        # Just compute the models if they are not already computed
        if not os.path.exists(os.path.join('results', exp_names[i])):
            # run main.py with subprocess
            command = f'python main.py --source {args.source} --all_vs_one {labels[i]} --exp_name {exp_names[i]} --batch_norm {args.batch_norm} --sample_frac {args.sample_frac} --weights {args.weights} --mode train'
            print(command)
            command = command.split()
            subprocess.call(command)



# Plot results
if args.mode == 'plot' or args.mode == 'compute':

    # Get results in validation ##########################################################################
    metric_paths = [os.path.join('results', exp_name, 'metric_dicts.pickle') for exp_name in exp_names]
    # Load metric dicts
    fold_performance_list = [pkl.load(open(metric_path, 'rb')) for metric_path in metric_paths]

    # Declare the dict that will have all the plotting information
    class_plot_dict = {}

    # Cycle over labels
    for i, name in enumerate(labels):
        # Obtain the fold performance dict of a given class
        class_fold_performance = fold_performance_list[i]
        # Obtain all the relevant metrics for this class
        class_performance_df = get_final_performance_df(class_fold_performance)

        # Declare an empty list that will contain the pr_df's of each fold
        pr_df_list = []
        pos_ap_list = []
        # Cycle over each fold to get PR curve
        for fold in class_fold_performance.keys():
            # Get the precision recall dataframe of the last epoch in that fold
            curr_fold = class_fold_performance[fold]
            final_pr_df = curr_fold['test'][-1]['pr_df']
            # Append pr_df to list
            pr_df_list.append(final_pr_df)
            # Append to positive AP list
            pos_ap_list.append(curr_fold['test'][-1]['AP_list'][1])
            # Compute the size of the training set. This will compute for all folds but the value used will only
            # be the last one. However all n's must be almost the same
            n = (curr_fold['train'][-1]['pr_df']['lab_num']==1).sum() + (curr_fold['test'][-1]['pr_df']['lab_num']==1).sum()
        
        # Concatenate the precision recall dataframes of each fold in a single one
        class_pr_df = pd.concat(pr_df_list, ignore_index=True)
        # Compute the precision recall curve for all data in a given class
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(class_pr_df['lab_num'], class_pr_df['positive_prob'])

        # Compute mean of positive AP
        mean_pos_ap = np.mean(pos_ap_list)

        # Add the information to the plotting dict
        class_plot_dict[name] = {'p': precision, 'r': recall, 't': thresholds, 'performance_df': class_performance_df, 'pos_ap': mean_pos_ap, 'pos_ap_list': pos_ap_list, 'n': n}


    # Compute lists of mAP and max_f1
    f1_list = [class_plot_dict[lab]['performance_df'].loc['Mean', 'max_f1'] for lab in labels]
    AP_list = [class_plot_dict[lab]['pos_ap'] for lab in labels]

    # This computes both the global mean performance and the standard deviation for maxF1 and mean AP
    # Get max F1 dataframe
    f1_df = pd.DataFrame({lab: class_plot_dict[lab]['performance_df'].loc[class_plot_dict[lab]['performance_df'].index.str.contains('Fold'), 'max_f1'] for lab in labels})
    # Get general mean and standard deviation of the means in each fold
    f1_glob_mean = f1_df.mean().mean()
    f1_std_bet_fold = f1_df.mean(axis=1).std()
    # Get positive AP dataframe
    ap_df = pd.DataFrame({lab: class_plot_dict[lab]['pos_ap_list'] for lab in labels})
    ap_df.index = [f'Fold {i+1}' for i in range(len(ap_df))]
    # Get general mean and standard deviation of the means in each fold
    ap_glob_mean = ap_df.mean().mean()
    ap_std_bet_fold = ap_df.mean(axis=1).std()


    # Handle colors to plot
    n_vec = np.array([class_plot_dict[lab]['n'] for lab in labels])
    normalization = matplotlib.colors.PowerNorm(gamma = 0.3, vmin=np.min(n_vec), vmax=np.max(n_vec))
    color_vec = normalization(n_vec)
    d_colors = ["black", "darkcyan"]
    cmap1 = LinearSegmentedColormap.from_list("cmap", d_colors)
    color_matrix = cmap1(color_vec)

    # This code performs the summary plot of the detection experiments
    f, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.87, 1]}, figsize=(23,10))
    f_scores = np.linspace(0.1, 0.9, num=9)
    for f_score in f_scores:
        x = np.linspace(0.01, 1.01, 499)
        y = f_score * x / (2 * x - f_score)
        ax[0].plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    for i, lab in enumerate(labels):
        ax[0].plot(class_plot_dict[lab]['r'], class_plot_dict[lab]['p'], color=color_matrix[i])
    
    ax[0].set_xlabel('Recall', fontsize=24)
    ax[0].set_ylabel('Precision', fontsize=24)
    ax[0].set_ylim([0.0, 1.01])
    ax[0].set_xlim([0.0, 1.01])
    ax[0].set_title('Precision-Recall Curve', fontsize=28)
    ax[0].grid(alpha=0.7)
    ax[0].tick_params(labelsize=15)
    ax[0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax[0].transAxes, size=20, weight='bold')
    plt.gca().set_axisbelow(True)


    # Plot of max F1 vs AP
    # plot max f1 vs ap. Set the size of the marker to be the size of the training set
    ax[1].scatter(AP_list, f1_list, s=1.5*n_vec, c=color_matrix, alpha=0.8)
    texts = [ax[1].text(AP_list[i]+n_vec[i]/40000, f1_list[i]+n_vec[i]/40000, labels[i]+' ({})'.format(n_vec[i]), ha='left', va='bottom') for i in range(len(labels)) if (AP_list[i]<0.8 or f1_list[i]<0.8)]
    [text.set_fontsize(13) for text in texts]
    adjust_text(texts)
    plt.xlim([0.0, 1.04])
    plt.ylim([0.0, 1.04])
    plt.xlabel('$AP$', fontsize=24)
    plt.ylabel('Max $F_1$', fontsize=24)
    plt.title('Max $F_1$ Vs Average Precision', fontsize=28)
    # Put box annotations on the plot
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax[1].text(0.30, 0.975, f"Average Max $F_1={round(f1_glob_mean,3)} \pm {round(f1_std_bet_fold,3)}$\nAverage $AP={round(ap_glob_mean,3)} \pm {round(ap_std_bet_fold,3)}$", ha="center", va="center", size=20, bbox=bbox_props)
    ax[1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax[1].transAxes, size=20, weight='bold')
    plt.tick_params(labelsize=15)
    plt.grid(alpha=0.7)
    plt.gca().set_axisbelow(True)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=cmap1), ax=ax[1])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Train Samples', fontsize=24)
    plt.tight_layout(w_pad=3)
    plt.show()
    plt.savefig(os.path.join('results',exp_folder_name, args.dataset,'pr_curves_summary.png'), dpi=400)
    plt.close()


