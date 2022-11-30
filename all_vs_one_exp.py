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

################ Parser code ###########################
# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--source',         type=str,       default="toil",     help="Data source to use",              choices=["toil", "wang","recount3"])
parser.add_argument('--dataset',        type=str,       default="tcga",     help="Dataset to use",                  choices=["tcga", "gtex", "both"])
parser.add_argument('--weights',        type=str,       default="True",     help="Whether to use weights in model", choices=["True", "False"])
parser.add_argument('--sample_frac',    type=float,     default=0.5,        help="Expression fraction threshold")
parser.add_argument('--mode',           type=str,       default="compute",  help="Mode to run the code.",           choices=["compute", "plot", "test"])
parser.add_argument('--gpu',            type=str,       default="0",          help="GPU on which to run experiments")
# Parse the argument
args = parser.parse_args()
#############################################################

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

with open(mapper_path, "r") as f:
    category_mapper = json.load(f)

labels = list(category_mapper.values())

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
        if not os.path.exists(os.path.join('Results', exp_names[i])):
            # run main.py with subprocess
            command = f'python main.py --source {args.source} --all_vs_one {labels[i]} --exp_name {exp_names[i]} --batch_norm normal --sample_frac {args.sample_frac} --weights {args.weights} --mode train'
            print(command)
            command = command.split()
            subprocess.call(command)

# Get results in validation ##########################################################################
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
normalization = matplotlib.colors.PowerNorm(gamma = 0.3, vmin=np.min(n_vec), vmax=np.max(n_vec))
color_vec = normalization(n_vec)
d_colors = ["black", "darkcyan"]
cmap1 = LinearSegmentedColormap.from_list("mycmap", d_colors)
color_matrix = cmap1(color_vec)

# Plot results
if args.mode == 'plot' or args.mode == 'compute':
    # TODO: This plot code can be way prettier
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
    plt.savefig(os.path.join('Results',exp_folder_name, args.dataset,'f1_ap_vs_samples.png'), dpi=200)
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
    ax[0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax[0].transAxes, size=20, weight='bold')
    plt.gca().set_axisbelow(True)

    # Plot of max F1 vs AP
    # plot max f1 vs ap. Set the size of the marker to be the size of the training set
    ax[1].scatter(AP_list, f1_list, s=2*np.array(n_list), c=color_matrix, alpha=0.8)
    texts = [ax[1].text(AP_list[i]+n_list[i]/40000, f1_list[i]+n_list[i]/40000, labels[i]+' ({})'.format(n_list[i]), ha='left', va='bottom') for i in range(len(labels)) if (AP_list[i]<0.8 or f1_list[i]<0.8)]
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
    ax[1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax[1].transAxes, size=20, weight='bold')
    plt.tick_params(labelsize=15)
    plt.grid(alpha=0.7)
    plt.gca().set_axisbelow(True)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=cmap1), ax=ax[1])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Train Samples', fontsize=24)
    plt.tight_layout(w_pad=3)
    plt.show()
    plt.savefig(os.path.join('Results',exp_folder_name, args.dataset,'pr_curves_summary.png'), dpi=400)
    plt.close()


# Get results in test ##########################################################################
# FIXME: Make possible to test with both wang and recount3
if args.mode == 'test':
    
    complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']
    pr_curve_list = []
    AP_list = []
    f1_list = []
    

    for i, lab in enumerate(labels):

        binary_dict = {label: 0 for label in complete_label_list}
        binary_dict[lab] = 1

        # Declare dataset to test models
        curr_dataset = ToilDataset(os.path.join("data", "toil_data"),
                            dataset = args.dataset,
                            tissue = 'all',
                            binary_dict=binary_dict,
                            mean_thr = -10.0,
                            std_thr = 0.0,
                            rand_frac = 1.0,
                            sample_frac=args.sample_frac,
                            gene_list_csv = 'None',
                            label_type = 'phenotype',
                            batch_normalization='normal',
                            partition_seed = 0,
                            force_compute = False)

        # Obtain test loader
        _, _, test_loader = curr_dataset.get_dataloaders(batch_size = 100)

        # Declare model
        device = torch.device("cuda") 
        model = MLP([len(curr_dataset.filtered_gene_list)], out_size = curr_dataset.num_classes ).to(device)
        # Declare path to load final model
        final_model_path = glob.glob(os.path.join('Results', exp_names[i], "checkpoint_epoch_*"))

        # Load final model dicts
        total_saved_dict = torch.load(final_model_path[0])
        model_dict = total_saved_dict['model_state_dict']

        # Load state dicts to model and optimizer
        model.load_state_dict(model_dict)

        # Put model in eval mode
        model.eval()

        # Obtain test metrics
        test_metrics = test(test_loader, model, device, num_classes=curr_dataset.num_classes)

    
        # Get val pr curve
        pr_curve = test_metrics['pr_curve']
        pr_curve_list.append(pr_curve)
        # Get val AP
        test_AP = test_metrics['AP_list'][1]
        AP_list.append(test_AP)
        # Get val max F1 score
        test_max_F1 = test_metrics['max_f1']
        f1_list.append(test_max_F1)
    
    # Plot test results
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
    ax[0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=ax[0].transAxes, size=20, weight='bold')
    plt.gca().set_axisbelow(True)

    # Plot of max F1 vs AP
    # plot max f1 vs ap. Set the size of the marker to be the size of the training set
    ax[1].scatter(AP_list, f1_list, s=2*np.array(n_list), c=color_matrix, alpha=0.8)
    texts = [ax[1].text(AP_list[i]+n_list[i]/40000, f1_list[i]+n_list[i]/40000, labels[i]+' ({})'.format(n_list[i]), ha='left', va='bottom') for i in range(len(labels)) if (AP_list[i]<0.8 or f1_list[i]<0.8)]
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
    ax[1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=ax[1].transAxes, size=20, weight='bold')
    plt.tick_params(labelsize=15)
    plt.grid(alpha=0.7)
    plt.gca().set_axisbelow(True)
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=cmap1), ax=ax[1])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Train Samples', fontsize=24)
    plt.tight_layout(w_pad=3)
    plt.show()
    plt.savefig(os.path.join('Results',exp_folder_name, args.dataset,'pr_curves_summary_test.png'), dpi=400)
    plt.close()
        


        





