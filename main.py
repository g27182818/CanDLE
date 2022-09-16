# Import of needed packages
import numpy as np
import os
import torch
import pickle
# Import auxiliary functions
from utils import *
from model import *
from datasets import *
# Set matplotlib option to plot while in screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set axis bellow for matplotlib
plt.rcParams['axes.axisbelow'] = True

################ Temporal parser code #######################
################ Must be replaced by configs #################
# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--dataset',        type=str,   default="both",         help="Dataset to use",                                                                                                  choices=["both", "tcga", "gtex"])
parser.add_argument('--tissue',         type=str,   default="all",          help="Tissue to use from data",                                                                                         choices=['all', 'Bladder', 'Blood', 'Brain', 'Breast', 'Cervix', 'Colon', 'Connective', 'Esophagus', 'Kidney', 'Liver', 'Lung', 'Not Paired', 'Ovary', 'Pancreas', 'Prostate', 'Skin', 'Stomach', 'Testis', 'Thyroid', 'Uterus'])
parser.add_argument('--all_vs_one',     type=str,   default='False',        help="If False solves a multiclass problem, if other string solves a binary problem with this as the positive class.",  choices=['False', 'GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM'])
parser.add_argument('--batch_norm',     type=str,   default="normal",       help="Normalization to perform in each subset of the dataset",                                                          choices=["none", "normal", "healthy_tcga"])
parser.add_argument('--seed',           type=int,   default=0,              help="Partition seed to divide tha data. Default is 0.")


parser.add_argument('--model',          type=str,   default="MLP_ALL",      help="Model to use. Baseline is a graph neural network",                                                                choices=["MLP_ALL", "MLP_FIL", "BASELINE"])

parser.add_argument('--lr',             type=float, default=0.00001,        help="Learning rate")
parser.add_argument('--batch_size',     type=int,   default=100,            help="Batch size")
parser.add_argument('--epochs',         type=int,   default=20,             help="Number of epochs")
parser.add_argument('--mode',           type=str,   default="train")
parser.add_argument('--train_samples',  type=int,   default=-1,             help='Number of samples used for training the algorithm. -1 to run with all data.') # TODO: Program subsampling in dataset. In this moment this still does not work
parser.add_argument('--exp_name',       type=str,   default='misc_test',    help="Experiment name to save")
# Parse the argument
args = parser.parse_args()
#############################################################

# ------------------- Important variable parameters -------------------------------------------------------------------#
# Miscellaneous parameters --------------------------------------------------------------------------------------------#
torch.manual_seed(12345)            # Set torch manual seed                                                            #
device = torch.device("cuda")       # Set cuda device                                                                  # # TODO: Make cuda or cpu if cuda is not available
mode = args.mode                    # Mode to run in code submission can be "test" or "demo"                           #
# Dataset parameters --------------------------------------------------------------------------------------------------#
mean_thr = -10.0                    # Mean expression threshold to filter out genes in toil.                           #  
std_thr = 0.0                       # Standard deviation threshold to filter out genes in toil.                        #
val_fraction = 0.2                  # Fraction of the data used for validation                                         #
train_smaples = args.train_samples  # Number of samples used for training the algorithm. -1 to run with all data.      #
dataset = args.dataset              # Dataset to use can be "both", "tcga" or "gtex"                                   #
tissue = args.tissue                # Tissue to use from data. "all" to use all tissues                                #
batch_size = args.batch_size        # Batch size parameter                                                             #
all_vs_one = args.all_vs_one        # If False multiclass problem else defines the positive class for binary problem   #
batch_norm = args.batch_norm        # Kind of normalization to perform in the subsets of data                          #
# Training parameters -------------------------------------------------------------------------------------------------#
experiment_name = args.exp_name     # Experiment name to define path were results are stored                           #
lr = args.lr                        # Learning rate of the Adam optimizer (was changed from 0.001 to 0.00001)          #
total_epochs = args.epochs          # Total number of epochs to train                                                  #
# ---------------------------------------------------------------------------------------------------------------------#

# Handle the posibility of an all vs one binary problem
complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']
if all_vs_one=='False':
    binary_dict = {}
else:
    binary_dict = {label: 0 for label in complete_label_list}
    binary_dict[all_vs_one] = 1

# Declare dataset
dataset = ToilDataset(os.path.join("data", "toil_data"),
                            dataset = dataset,
                            tissue = tissue,
                            binary_dict=binary_dict,
                            mean_thr = mean_thr,
                            std_thr = std_thr,
                            label_type = 'phenotype',
                            batch_normalization=batch_norm,
                            partition_seed = args.seed,
                            force_compute = False)

# Dataloader declaration
train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size = batch_size)


# Calculate loss function weights
distribution = np.bincount(np.ravel(dataset.split_labels["train_val"].values).astype(np.int64))
loss_wieghts = 2500000 / (distribution**2)
lw_tensor = torch.tensor(loss_wieghts, dtype=torch.float).to(device)

# Model definition
model = MLP([len(dataset.filtered_gene_list)], out_size = dataset.num_classes ).to(device)
# Print to console model definition
print("The model definition is:")
print(model)

# Define optimizer and criterion
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
criterion = torch.nn.CrossEntropyLoss(weight=lw_tensor) # TODO: Put loss weights by parameter
# criterion = torch.nn.CrossEntropyLoss() # Temporal experiment without weights

# Lists declarations
train_metric_lst = []
val_metric_lst = []
loss_list = []

# Declare results path
# TODO: Make function that returns all the paths
results_path = os.path.join("Results", experiment_name)
# Declare log path
train_log_path = os.path.join(results_path, "TRAINING_LOG.txt")
# Declare metric dicts path
metrics_log_path = os.path.join(results_path, "metric_dicts.pickle")
# Declare path to save performance training plot
train_performance_fig_path = os.path.join(results_path, "training_performance.png")
# Declare path to save final confusion matrices
conf_matrix_fig_path = os.path.join(results_path, "confusion_matrix")
# Declare path to save pr curves for binary classification
pr_curves_fig_path = os.path.join(results_path, "pr_curves.png")

# Create results directory
if not os.path.isdir(results_path):
    os.makedirs(results_path)

if mode == "train":
    # Train/test cycle
    for epoch in range(total_epochs):
        print('-----------------------------------------')
        print("Epoch " + str(epoch+1) + ":")
        print('                                         ')
        print("Start training:")

        # Train one epoch
        loss = train(train_loader, model, device, criterion, optimizer)

        # Obtain test metrics for each epoch in all groups
        print('                                         ')
        print("Obtaining train metrics:")
        train_metrics = test(train_loader, model, device, num_classes=dataset.num_classes)

        print('                                         ')
        print("Obtaining val metrics:")
        val_metrics = test(val_loader, model, device, num_classes=dataset.num_classes)

        # Add epoch information to the dictionaries
        train_metrics["epoch"] = epoch
        val_metrics["epoch"] = epoch

        # Append data to list
        train_metric_lst.append(train_metrics)
        val_metric_lst.append(val_metrics)
        loss_list.append(loss.cpu().detach().numpy())

        # Print performance
        print_epoch(train_metrics, val_metrics, loss, epoch, train_log_path)

        # Save checkpoint at last epoch
        if epoch+1== total_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss},
                os.path.join(results_path, "checkpoint_epoch_"+str(epoch+1)+".pt"))

    # Save metrics dicts
    complete_metric_dict = {"train": train_metric_lst,
                            "val": val_metric_lst,
                            "loss": loss_list}

    with open(metrics_log_path, 'wb') as f:
        pickle.dump(complete_metric_dict, f)

    # Generate training performance plot and save it to train_performance_fig_path
    plot_training(train_metric_lst, val_metric_lst, loss_list, train_performance_fig_path)

    # Plot PR curve if the problem is binary
    if dataset.num_classes == 2:
        plot_pr_curve(train_metrics["pr_curve"], val_metrics["pr_curve"], pr_curves_fig_path)
                      
    # Generate confusion matrices plot and save it to conf_matrix_fig_path
    plot_conf_matrix(train_metrics["conf_matrix"],
                        val_metrics["conf_matrix"],
                        dataset.lab_txt_2_lab_num,
                        conf_matrix_fig_path)

elif mode == 'test':
    # Declare path to load final model
    final_model_path = os.path.join(results_path, "checkpoint_epoch_"+str(total_epochs)+".pt")
    # Declare path to save gene ranking csv
    gene_ranking_path = os.path.join(results_path, "one_candle_gene_ranking.csv")

    # Load final model dicts
    total_saved_dict = torch.load(final_model_path)
    model_dict = total_saved_dict['model_state_dict']
    optimizer_dict = total_saved_dict['optimizer_state_dict']

    # Load state dicts to model and optimizer
    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optimizer_dict)

    # Put model in eval mode
    model.eval()

    # Obtain test metrics
    test_metrics = test(test_loader, model, device, num_classes=dataset.num_classes)
    # Print metrics
    print('The metrics before the attack in test set are:')
    print('balanced accuracy = {}'.format(test_metrics['mean_acc']))
    print('total accuracy = {}'.format(test_metrics['tot_acc']))
    print('mean average precision = {}'.format(test_metrics['mean_AP']))


    # This is the code for the interpretation of one model of candle
    # Get model weights 
    weight_matrix = model.out.weight.detach().cpu().numpy()
    tcga_weight_matrix = weight_matrix[30:, :]

    # Code to sort gene weights for each class
    gene_names = np.array(dataset.filtered_gene_list) # Get original gene names
    rankings = np.argsort(np.abs(tcga_weight_matrix)) # Obtain rankings based on the absolute value of W
    
    # Declare empty matrix to sort TCGA weights 
    sorted_tcga_weight_matrix = np.zeros_like(tcga_weight_matrix)

    # Cycle to assign sorted weights and order rankings from biggest to lower absolute value
    for i in range(len(sorted_tcga_weight_matrix)):
        sorted_vec = tcga_weight_matrix[i, rankings[i][::-1]]
        sorted_tcga_weight_matrix[i] = sorted_vec
        rankings[i] = rankings[i][::-1]
    
    # Number of genes to be selected as important predictors for each cancer class
    k = 1000

    # Get the top-k important genes in each cancer class
    top_k_ranking = rankings[:, :k]
    # Count the number of times each gene was selected in the top-k for any cancer class
    frecuencies = np.bincount(top_k_ranking.flatten())
    
    # Obtain the ranking of frequencies and sort frequency vector
    frec_rank = np.argsort(frecuencies)[::-1]
    gene_frec_sorted = gene_names[frec_rank]
    frecuencies_sorted = frecuencies[frec_rank]

    # Make a datarfame of interpretation results, print it and save it to file
    rank_frec_df = pd.DataFrame({'gene_name': gene_frec_sorted, 'frec': frecuencies_sorted})
    print(rank_frec_df)
    pd.DataFrame(rank_frec_df).to_csv('one_candle_weights.csv')

    # Make scatter plot of weights of a single random class
    rand_int = np.random.randint(0,33)
    threshold = sorted_tcga_weight_matrix[rand_int, k]
    plt.figure()
    plt.plot(tcga_weight_matrix[i], '.k', markersize=2, alpha=0.4)
    plt.plot([0, len(tcga_weight_matrix[i])],[threshold, threshold], '--r')
    plt.plot([0, len(tcga_weight_matrix[i])],[-threshold, -threshold], '--r')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, len(tcga_weight_matrix[i]))
    plt.xlabel('Gene', fontsize=16)
    plt.ylabel('$w ($Gene$)$', fontsize='large')
    plt.title(f'Weights for class {rand_int}', fontsize='xx-large')
    plt.show()
    plt.tight_layout()
    plt.savefig('random_class_weights_plot.png', dpi=300)
