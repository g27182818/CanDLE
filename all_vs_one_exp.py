import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import pickle as pkl
import matplotlib

mode = 'compute' # 'compute' or 'plot'
dataset = 'both' # 'tcga', 'gtex' or 'both'
gpu = '2'

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

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


# # Temporal test sumsample
# labels = labels[:21]

# Make dir for results
exp_names = [os.path.join('all_vs_one_exp', dataset, label) for label in labels]

if mode == 'compute':
    for i in range(len(labels)):
        # run main.py with subprocess
        command = 'python main.py --all_vs_one {} --exp_name {}'.format(labels[i], exp_names[i])
        print(command)
        command = command.split()
        subprocess.call(command)
elif mode == 'plot' or mode == 'compute':
    metric_paths = [os.path.join('Results', exp_name, 'metric_dicts.pickle') for exp_name in exp_names]
    # Load metric dicts
    metric_dicts = [pkl.load(open(metric_path, 'rb')) for metric_path in metric_paths]

    pr_curve_list = []
    n_list = []
    for metric_dict in metric_dicts:
        # Get pr curve
        val_metrics = metric_dict['val'][-1]
        pr_curve = val_metrics['pr_curve']
        pr_curve_list.append(pr_curve)
        # Get sample number of training set
        train_metrics = metric_dict['train'][-1]
        conf_matrix_train = train_metrics['conf_matrix']
        n_train = np.sum(conf_matrix_train[1,:])
        n_list.append(n_train)

    # Handle colors to plot
    n_vec = np.array(n_list)
    # color_vec = (n_vec-np.min(n_vec)) / (np.max(n_vec)-np.min(n_vec))
    normalization = matplotlib.colors.LogNorm(vmin=np.min(n_vec), vmax=np.max(n_vec))
    color_vec = normalization(n_vec)
    color_matrix = plt.cm.magma(color_vec)
    

    plt.figure(figsize=(13,10))
    f_scores = np.linspace(0.1, 0.9, num=9)
    for f_score in f_scores:
        x = np.linspace(0.01, 1.01, 499)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    for i in range(len(pr_curve_list)):
        pr_curve = pr_curve_list[i]
        plt.plot(pr_curve[1], pr_curve[0], color=color_matrix[i])
    plt.xlabel('Recall', fontsize=24)
    plt.ylabel('Precision', fontsize=24)
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    plt.title('Precision-Recall Curve', fontsize=28)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap='magma'))
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Train Samples', fontsize=24)
    plt.grid()
    plt.tight_layout()
    plt.tick_params(labelsize=15)
    plt.savefig(os.path.join('all_vs_one_exp', dataset,'joint_pr_curve.png'), dpi=200)
    plt.close()

else:
    raise ValueError('mode must be either "compute" or "plot"')


