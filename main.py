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
from dataloader import *
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
experiment_name = "lr0.00001_TCGA_no_filter"  # Experiment name to define path were results are stored                           #
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
    mean_thr = 0.5  # Mean threshold for filtering input genes
    std_thr = 0.8  # Standard deviation threshold for filtering input genes
    # Obtain data matrices (in numpy) from dataloader funciton. X and Y are already shuffled
    X_np, Y_np, ID, loc2gene = matrixloader(mean_thr, std_thr)
    # Obtain edges and edges atributes, it assigns empty tensors.
    edge_indices, edge_attributes = torch.empty((2, 1)), torch.empty((1,))

elif model_type == "BASELINE":
    mean_thr = 0.5  # Mean threshold for filtering input genes
    std_thr = 0.8  # Standard deviation threshold for filtering input genes
    # Obtain data matrices (in numpy) from dataloader funciton. X and Y are already shuffled
    X_np, Y_np, ID, loc2gene = matrixloader(mean_thr, std_thr)
    # Obtain edges and edges atributes
    edge_indices, edge_attributes = generate_graph_adjacency(X_np, coor_thr, p_value_thr)

elif model_type == "MLP_ALL":
    mean_thr = 0.0  # Mean threshold for filtering input genes
    std_thr = 0.0  # Standard deviation threshold for filtering input genes
    ########################################################################################
    # TEMPORAL CODE MADE TO INTEGRATE GTEX DATA ############################################
    ########################################################################################
    compute = True
    gtex = False
    filter_intersection = False

    if compute ==True:
        print('Started reading TCGA data...')
        # Obtain data matrices (in numpy) from dataloader funciton. X and Y are already shuffled
        X_tcga, Y_tcga, ID, loc2gene = matrixloader(mean_thr, std_thr)
        
        # Remove normal from TCGA if gtex is used
        if gtex == True:
            print('Modifing TCGA data...')
            # Remove normal TCGA dada
            remove_normal_index_tcga = Y_tcga != 0
            Y_tcga = Y_tcga[remove_normal_index_tcga]
            X_tcga = X_tcga[remove_normal_index_tcga, :]
            # Lower annotations one unit (due to removal of Normal TCGA tissue) 
            Y_tcga = Y_tcga - 1

        if filter_intersection == True:
            print('Filtering TCGA data...')
            # Compute dictionary mapping each gene name to column index in X_tcga
            gene2loc = dict((v, k) for k, v in loc2gene.items())
            df_gene2loc = pd.DataFrame.from_dict(gene2loc, orient='index')
            
            # Read intersection genes
            intersection_genes = pd.read_pickle('gene_intersection.pkl')

            # Unexpected new intersection being performed
            intersection_list = list(set(intersection_genes.tolist())&set(df_gene2loc.index))
            intersection_list = sorted(intersection_list)

            # Find locations of intersection genes in X_tcga
            locations = df_gene2loc.loc[intersection_list]

            # Filter X_tcga in intersection genes locations
            X_tcga = X_tcga[:, np.ravel(locations.to_numpy())]

        print('Performing transformations on TCGA data...')
        # Reverse log2 transform
        X_tcga = np.exp2(X_tcga)-1
        # Put X_tcga in TPM normalization
        X_tcga = 1e6 * X_tcga/np.sum(X_tcga, axis=0, keepdims=True)
        # Perform log2 normalization on X_tcga
        X_tcga = np.log2(X_tcga+1)

        if gtex == True:
            print('Started reading GTEx data...')
            ordered_genes_gtex = pd.read_pickle(os.path.join('processed_gtex', 'ordered_genes.pkl'))
            # Read X_gtex and Y_gtex data
            X_gtex = np.load(os.path.join('processed_gtex', 'gtex_x.npy'), allow_pickle=True).astype(np.float64)
            Y_gtex = np.load(os.path.join('processed_gtex', 'gtex_y.npy'), allow_pickle=True)

            print('Filtering GTEx data...')
            # Filter X_gtex with intersection list the same was that was done with X_tcga
            gene2loc_gtex = {ordered_genes_gtex[i]:i for i in range(len(ordered_genes_gtex))}
            df_gene2loc_gtex = pd.DataFrame.from_dict(gene2loc_gtex, orient='index')
            locations_gtex = df_gene2loc_gtex.loc[intersection_list]
            X_gtex = X_gtex[:, np.ravel(locations_gtex.to_numpy())]

            print('Transforming GTEx data...')
            X_gtex = np.log2(X_gtex+1)

            # Create unified X_np matrix
            print('Merging TCGA and GTEx data...')
            X_np = np.vstack((X_tcga, X_gtex))
            Y_np = np.concatenate((Y_tcga, Y_gtex))
        
        else:
            X_np = X_tcga
            Y_np = Y_tcga

        # Specify seed for shuffle
        print('Shuffling complete data...')
        np.random.seed(1)
        shuffler = np.random.permutation(len(X_np))
        X_np = X_np[shuffler]
        Y_np = Y_np[shuffler]

        # # Create save directories
        # try:
        #     os.makedirs(os.path.join('final_dataset'))
        # except:
        #     print(os.path.join('final_dataset')+" directory already exist")
        
        # # Save X_np, Y_np and intersection_list
        # print('Saving total X to file...')
        # with open(os.path.join('final_dataset', 'X.pkl'), 'wb') as f:
        #     pickle.dump(X_np, f)
        # print('Saving total Y to file...')
        # with open(os.path.join('final_dataset', 'Y.pkl'), 'wb') as f:
        #     pickle.dump(Y_np, f)
        # print('Saving intersection genes to file...')
        # with open(os.path.join('final_dataset', 'intersection_genes.pkl'), 'wb') as f:
        #     pickle.dump(intersection_list, f)
    else:
        # Load important variables from files
        X_np = pd.read_pickle(os.path.join('final_dataset', 'X.pkl'))
        Y_np = pd.read_pickle(os.path.join('final_dataset', 'Y.pkl'))
        intersection_list = pd.read_pickle(os.path.join('final_dataset', 'intersection_genes.pkl'))

    # breakpoint()

    #########################################################################################
    #########################################################################################
    #########################################################################################
    # Obtain edges and edges atrubutes, it assigns empty tensors.
    edge_indices, edge_attributes = torch.empty((2, 1)), torch.empty((1,))

else:
    raise NotImplementedError

# Define number of classes
classes_number = int(np.max(Y_np)+1)

# Assign train and validation data in numpy
X_val_np = X_np[int((1-val_fraction)*X_np.shape[0]):, :]
Y_val_np = Y_np[int((1-val_fraction)*X_np.shape[0]):]

X_train_np = X_np[0:int((1-val_fraction)*X_np.shape[0]), :]
Y_train_np = Y_np[0:int((1-val_fraction)*X_np.shape[0])]

# Handle possible subsampling of train set
if (train_smaples > -1) & (train_smaples < X_train_np.shape[0]-np.max(Y_val_np)):
    X_train_np, _, Y_train_np, _ = train_test_split(X_train_np, Y_train_np,
                                                    stratify=Y_train_np,
                                                    train_size=train_smaples,
                                                    random_state=0)

# Transform to tensors train data
X_train = torch.tensor(X_train_np, dtype=torch.float)
Y_train = torch.tensor(Y_train_np)

# Transform to tensors validation data
X_val = torch.tensor(X_val_np, dtype=torch.float)
Y_val = torch.tensor(Y_val_np)

# Define datalists of graphs
train_graph_list = [Data(x=torch.unsqueeze(X_train[i, :], 1),
                         y=Y_train[i],
                         edge_index=edge_indices,
                         edge_attributes=edge_attributes,
                         num_nodes=len(X_train[i, :])) for i in range(X_train.shape[0])]

val_graph_list = [Data(x=torch.unsqueeze(X_val[i, :], 1),
                       y=Y_val[i],
                       edge_index=edge_indices,
                       edge_attributes=edge_attributes,
                       num_nodes=len(X_val[i, :])) for i in range(X_val.shape[0])]

# Dataloader declaration
train_loader = DataLoader(train_graph_list, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graph_list, batch_size=batch_size)

# Calculate loss function weights
distribution = np.bincount(np.ravel(Y_np).astype(np.int64))
loss_wieghts = 200 / distribution
lw_tensor = torch.tensor(loss_wieghts, dtype=torch.float).to(device)

# Handle model definition based on model type
if model_type == "MLP_FIL" or model_type == "MLP_ALL":
    model = MLP([X_np.shape[1]], out_size = classes_number ).to(device)
elif model_type == "BASELINE":
    model = BaselineModel(hidden_channels=hidd, input_size=X_np.shape[1], out_size=classes_number).to(device)
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
        train_metrics = test(train_loader, model, device, metric, num_classes=classes_number)

        print('                                         ')
        print("Obtaining test metrics:")
        test_metrics = test(val_loader, model, device, metric, num_classes=classes_number)

        # Handle if adversarial testing is required
        if test_adversarial:
            print('                                         ')
            print("Obtaining adversarial test metrics:")
            # This test is set to use 50 iterations of APGD
            adv_test_metrics = test(val_loader, model, device, metric,
                                    optimizer=optimizer, adversarial=True,
                                    attack=apgd_graph, criterion=criterion,
                                    epsilon=test_eps, n_iter=50,
                                    num_classes=classes_number)

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
                         conf_matrix_fig_path)
elif mode == "demo":
    demo_saved_dict = torch.load("best_model.pt")
    model_dict = demo_saved_dict['model_state_dict']
    # Load state dicts to model and optimizer
    model.load_state_dict(model_dict)
    # Make demo loader with unit size of batch
    demo_loader = DataLoader(val_graph_list, batch_size=1)
    # Make demo plot
    demo(demo_loader, model, device, num_test)
else:
    raise NotImplementedError
