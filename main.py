# Import of needed packages
import numpy as np
from sklearn.model_selection import train_test_split
import time
import scipy.io as sio
import os
import sys
import torch
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import tqdm
import pandas as pd
# Import auxiliary functions
from utils import *
from model import *
from datasets import *
# Set matplotlib option to plot while in screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

################ Temporal parser code #######################
################ Must be replace by configs #################
# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--adv_e_test', type=float, default=0.01)
parser.add_argument('--adv_e_train', type=float, default=0.00)
parser.add_argument('--n_iters_apgd', type=int, default=50)
parser.add_argument('--mode', type=str, default="test")
parser.add_argument('--num_test', type=int, default=69)
parser.add_argument('--train_samples', type=int, default=-1)
# Parse the argument
args = parser.parse_args()
#############################################################

# ------------------- Important variable parameters -------------------------------------------------------------------#
# Miscellaneous parameters --------------------------------------------------------------------------------------------#
torch.manual_seed(12345)            # Set torch manual seed                                                            #
device = torch.device("cuda")       # Set cuda device                                                                  #
mode = args.mode                    # Mode to run in code submission can be "test" or "demo"                           #
num_test = args.num_test            # Number of demo data to plot                                                      #
# Dataset parameters --------------------------------------------------------------------------------------------------#
val_fraction = 0.2                  # Fraction of the data used for validation                                         #
train_smaples = args.train_samples  # Number of samples used for training the algorithm. -1 to run with all data.      #
batch_size = 100                    # Batch size parameter                                                             #
coor_thr = 0.6                      # Spearman correlation threshold for declaring graph topology                      #
p_value_thr = 0.05                  # P-value Spearman correlation threshold for declaring graph topology              #
# Model parameters ----------------------------------------------------------------------------------------------------#
hidd = 8                            # Hidden channels parameter for baseline model                                     #
model_type = "MLP_ALL"              # Model type, can be "MLP_FIL", "MLP_ALL", "BASELINE"                              #
# Training parameters -------------------------------------------------------------------------------------------------#
experiment_name = "test_toil_just_gtex"       # Experiment name to define path were results are stored                           #
lr = 0.00001                        # Learning rate of the Adam optimizer (was changed from 0.001 to 0.00001)          #
total_epochs = 20                   # Total number of epochs to train                                                  #
metric = 'both'                     # Evaluation metric for experiment. Can be 'acc', 'mAP' or 'both'                  #
train_eps = args.adv_e_train        # Adversarial epsilon for train                                                    #
n_iters_apgd = args.n_iters_apgd    # Number of performed APGD iterations in train                                     #
# Test parameters -----------------------------------------------------------------------------------------------------#
test_eps = args.adv_e_test          # Adversarial epsilon for test                                                     #
# ---------------------------------------------------------------------------------------------------------------------#

# Handle input filtering depending in model type
if model_type == "MLP_FIL":
    mean_thr = 3.0  # Mean threshold for filtering input genes
    std_thr = 0.5   # Standard deviation threshold for filtering input genes
    use_graph = False
elif model_type == "BASELINE":
    mean_thr = 3.0  # Mean threshold for filtering input genes
    std_thr = 0.5   # Standard deviation threshold for filtering input genes
    use_graph = True
elif model_type == "MLP_ALL":
    mean_thr = -10.0  # Mean threshold for filtering input genes
    std_thr = -1.0    # Standard deviation threshold for filtering input genes
    use_graph = False
else:
    raise NotImplementedError


dataset = ToilDataset(os.path.join("data", "toil_data"),
                            tcga = False,
                            gtex = True,
                            mean_thr = mean_thr,
                            std_thr = std_thr,
                            use_graph = use_graph,
                            corr_thr = coor_thr,
                            p_thr = p_value_thr,
                            label_type = 'phenotype',
                            force_compute = False)

# Dataloader declaration
train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size = batch_size)


# Calculate loss function weights
distribution = np.bincount(np.ravel(dataset.split_labels["train_val"].values).astype(np.int64))
loss_wieghts = 200 / distribution
lw_tensor = torch.tensor(loss_wieghts, dtype=torch.float).to(device)

# Handle model definition based on model type
if model_type == "MLP_FIL" or model_type == "MLP_ALL":
    model = MLP([len(dataset.filtered_gene_list)], out_size = dataset.num_classes ).to(device)
elif model_type == "BASELINE":
    model = BaselineModel(hidden_channels=hidd, input_size=len(dataset.filtered_gene_list), out_size=dataset.num_classes).to(device)
else:
    raise NotImplementedError

# Print to console model definition
print("The model definition is:")
print(model)

# Define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
criterion = torch.nn.CrossEntropyLoss(weight=lw_tensor)

# Decide whether to train and test adversarially or not
train_adversarial = train_eps > 0.0
test_adversarial = train_eps > 0.0

# Lists declarations
train_metric_lst = []
test_metric_lst = []
adv_test_metric_lst = []
loss_list = []

# Declare results path
# results_path = os.path.join("Results", "MULTINOMIAL_LOGISTIC_ALL_DATA", "ADVERSARIAL_APGD_"+str(int(100*adv_e_train))+"%")
# results_path = os.path.join("Results", "TRAIN_SAMPLES_SWEEP", experiment_name)
results_path = os.path.join("Results", experiment_name)
# Declare log path
train_log_path = os.path.join(results_path, "TRAINING_LOG.txt")
# Declare metric dicts path
metrics_log_path = os.path.join(results_path, "metric_dicts.pickle")
# Declare path to save performance training plot
train_performance_fig_path = os.path.join(results_path, "training_performance.png")
# Declare path to save final confusion matrices
conf_matrix_fig_path = os.path.join(results_path, "confusion_matrix")

# Create results directory
if not os.path.isdir(results_path):
    os.makedirs(results_path)

if mode == "test":
    # Train/test cycle
    for epoch in range(total_epochs):
        print('-----------------------------------------')
        print("Epoch " + str(epoch+1) + ":")
        print('                                         ')
        print("Start training:")
        # Train one epoch adversarially
        if train_adversarial:
            loss = train(train_loader, model, device, criterion, optimizer,
                         adversarial=True, attack=apgd_graph, epsilon=train_eps,
                         n_iter=n_iters_apgd)
        # Train one epoch normally
        else:
            loss = train(train_loader, model, device, criterion, optimizer)

        # Obtain test metrics for each epoch in all groups
        print('                                         ')
        print("Obtaining train metrics:")
        train_metrics = test(train_loader, model, device, metric, num_classes=dataset.num_classes)

        print('                                         ')
        print("Obtaining test metrics:")
        test_metrics = test(val_loader, model, device, metric, num_classes=dataset.num_classes)

        # Handle if adversarial testing is required
        if test_adversarial:
            print('                                         ')
            print("Obtaining adversarial test metrics:")
            # This test is set to use 50 iterations of APGD
            adv_test_metrics = test(val_loader, model, device, metric,
                                    optimizer=optimizer, adversarial=True,
                                    attack=apgd_graph, criterion=criterion,
                                    epsilon=test_eps, n_iter=50,
                                    num_classes=dataset.num_classes)

        # If adversarial testing is not required adversarial test metrics are the same normal metrics
        else:
            adv_test_metrics = test_metrics

        # Add epoch information to the dictionaries
        train_metrics["epoch"] = epoch
        test_metrics["epoch"] = epoch
        adv_test_metrics["epoch"] = epoch

        # Append data to list
        train_metric_lst.append(train_metrics)
        test_metric_lst.append(test_metrics)
        adv_test_metric_lst.append(adv_test_metrics)
        loss_list.append(loss.cpu().detach().numpy())

        # Print performance
        print_epoch(train_metrics, test_metrics, adv_test_metrics, loss, epoch, train_log_path)

        # Save checkpoints every 2 epochs
        if (epoch+1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss},
                os.path.join(results_path, "checkpoint_epoch_"+str(epoch+1)+".pt"))

    # Save metrics dicts
    complete_metric_dict = {"train": train_metric_lst,
                            "test": test_metric_lst,
                            "adv_test": adv_test_metric_lst,
                            "loss": loss_list}

    with open(metrics_log_path, 'wb') as f:
        pickle.dump(complete_metric_dict, f)

    # Generate training performance plot and save it to train_performance_fig_path
    plot_training(metric, train_metric_lst, test_metric_lst, adv_test_metric_lst, loss_list, train_performance_fig_path)

    # Generate confusion matrices plot and save it to conf_matrix_fig_path
    if (metric == 'acc') or (metric == 'both'):
        plot_conf_matrix(train_metrics["conf_matrix"],
                         test_metrics["conf_matrix"],
                         adv_test_metrics["conf_matrix"],
                         dataset.lab_txt_2_lab_num,
                         conf_matrix_fig_path)
# TODO: Handle demo mode correctly
# elif mode == "demo":
#     demo_saved_dict = torch.load("best_model.pt")
#     model_dict = demo_saved_dict['model_state_dict']
#     # Load state dicts to model and optimizer
#     model.load_state_dict(model_dict)
#     # Make demo loader with unit size of batch
#     demo_loader = DataLoader(val_graph_list, batch_size=1)
#     # Make demo plot
#     demo(demo_loader, model, device, num_test)
else:
    raise NotImplementedError