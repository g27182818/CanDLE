import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle as pkl
import matplotlib
from adjustText import adjust_text

######################################################################
#            You can safely change these parameters                  #
######################################################################
mode = 'compute' # 'compute' or 'plot'
dataset = 'tcga' # 'tcga', 'gtex' or 'both'
use_weights = 'True' # 'True' or 'False'

gpu = '0' # What GPU to use
######################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

if use_weights=='True':
    exp_folder_name = 'CanDLE_all_vs_one_exp_1_epoch'
else:
    exp_folder_name = 'CanDLE_all_vs_one_exp_1_epoch_no_weights'

# Get mapper file to know labels
mapper_path = os.path.join('data','toil_data', 'mappers', 'category_mapper.json')
with open(mapper_path, "r") as f:
    category_mapper = json.load(f)

labels = list(category_mapper.values())

if dataset == 'both':
    pass
elif dataset == 'tcga':
    # Delete all label that start with 'GTEX'
    labels = [label for label in labels if not label.startswith('GTEX')]
elif dataset == 'gtex':
    # Delete all label that start with 'TCGA'
    labels = [label for label in labels if not label.startswith('TCGA')]
else:
    raise ValueError('dataset must be either "tcga", "gtex" or "both"')


# Declare folder names for all experiments
exp_names = [os.path.join(exp_folder_name, dataset, label) for label in labels]


# If mode is 'compute', compute all experiments
if mode == 'compute':
    for i in range(len(labels)):
        # Just compute the models if they are not already computed
        if not os.path.exists(os.path.join('Results', exp_names[i])):
            # run main.py with subprocess
            command = f'python main.py --all_vs_one {labels[i]} --exp_name {exp_names[i]} --batch_norm normal --std_thr 0.1 --weights {use_weights} --epochs 1'
            print(command)
            command = command.split()
            subprocess.call(command)

# Plot results
if mode == 'plot' or mode == 'compute':
    metric_paths = [os.path.join('Results', exp_name, 'metric_dicts.pickle') for exp_name in exp_names]
    # Load metric dicts
    metric_dicts = [pkl.load(open(metric_path, 'rb')) for metric_path in metric_paths]

    pr_curve_list = []
    n_list = []
    AP_list = []
    f1_list = []
    for metric_dict in metric_dicts:
        # Get val pr curve
        val_metrics = metric_dict['val'][-1]
        pr_curve = val_metrics['pr_curve']
        pr_curve_list.append(pr_curve)
        # Get val AP
        val_AP = val_metrics['AP_list'][1]
        AP_list.append(val_AP)
        # Get val max F1 score
        val_max_F1 = val_metrics['max_f1']
        f1_list.append(val_max_F1)
        # Get sample number of training set
        train_metrics = metric_dict['train'][-1]
        conf_matrix_train = train_metrics['conf_matrix']
        n_train = np.sum(conf_matrix_train[1,:])
        n_list.append(n_train)

    # Handle colors to plot
    n_vec = np.array(n_list)
    normalization = matplotlib.colors.LogNorm(vmin=np.min(n_vec), vmax=np.max(n_vec))
    color_vec = normalization(n_vec)
    color_matrix = plt.cm.magma(color_vec)


    plt.figure(figsize=(22,10))
    plt.subplot(121)
    plt.plot(n_list, AP_list, 'ok')
    plt.xlim([0, 1.01*max(n_list)])
    plt.ylim([0.5, 1.01])
    plt.xlabel('Train Samples', fontsize=24)
    plt.ylabel('$AP$', fontsize=24)
    plt.title('Average Precision Vs Training Samples', fontsize=28)
    plt.tick_params(labelsize=15)
    plt.grid()
    plt.subplot(122)
    plt.plot(n_list, f1_list, 'ok')
    plt.xlim([0, 1.01*max(n_list)])
    plt.ylim([0.5, 1.01])
    plt.xlabel('Train Samples', fontsize=24)
    plt.ylabel('Max $F_1$', fontsize=24)
    plt.title('Max $F_1$ Vs Training Samples', fontsize=28)
    plt.tick_params(labelsize=15)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join('Results',exp_folder_name, dataset,'f1_ap_vs_samples.png'), dpi=200)
    plt.close()


    f, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.87, 1]}, figsize=(23,10))
    f_scores = np.linspace(0.1, 0.9, num=9)
    for f_score in f_scores:
        x = np.linspace(0.01, 1.01, 499)
        y = f_score * x / (2 * x - f_score)
        ax[0].plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    for i in range(len(pr_curve_list)):
        pr_curve = pr_curve_list[i]
        ax[0].plot(pr_curve[1], pr_curve[0], color=color_matrix[i])
    
    ax[0].set_xlabel('Recall', fontsize=24)
    ax[0].set_ylabel('Precision', fontsize=24)
    ax[0].set_ylim([0.0, 1.01])
    ax[0].set_xlim([0.0, 1.01])
    ax[0].set_title('Precision-Recall Curve', fontsize=28)
    ax[0].grid(alpha=0.7)
    ax[0].tick_params(labelsize=15)
    plt.gca().set_axisbelow(True)

    # Plot of max F1 vs AP
    # plot max f1 vs ap. Set the size of the marker to be the size of the training set
    ax[1].scatter(AP_list, f1_list, s=2*np.array(n_list), c=color_matrix, alpha=0.8)
    texts = [ax[1].text(AP_list[i]+n_list[i]/40000, f1_list[i]+n_list[i]/40000, labels[i]+' ({})'.format(n_list[i]), ha='left', va='bottom') for i in range(len(labels)) if (AP_list[i]<0.6 or f1_list[i]<0.6)]
    [text.set_fontsize(13) for text in texts]
    adjust_text(texts)
    plt.xlim([0.0, 1.04])
    plt.ylim([0.0, 1.04])
    plt.xlabel('$AP$', fontsize=24)
    plt.ylabel('Max $F_1$', fontsize=24)
    plt.title('Max $F_1$ Vs Average Precision', fontsize=28)
    # Put box annotations on the plot
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    ax[1].text(0.25, 0.975, "Average Max $F_1={}$\nAverage $AP={}$".format(round(np.mean(f1_list),3), round(np.mean(AP_list),3)), ha="center", va="center", size=20,
            bbox=bbox_props)
    # plt.xticks(np.arange(0.55, 1.05, 0.05))
    # plt.yticks(np.arange(0.55, 1.05, 0.05))
    plt.tick_params(labelsize=15)
    plt.grid(alpha=0.7)
    plt.gca().set_axisbelow(True)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap='magma'), ax=ax[1])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Train Samples', fontsize=24)
    plt.tight_layout(w_pad=3)
    plt.show()
    plt.savefig(os.path.join('Results',exp_folder_name, dataset,'pr_curves_summary.png'), dpi=400)
    plt.close()






