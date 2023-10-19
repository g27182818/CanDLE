# Import of needed packages
import numpy as np
import os
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from thundersvm import SVC
# Import auxiliary functions
from utils import *
from model import *
from datasets import *
from metrics import get_metrics
# Set matplotlib option to plot while in screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Set axis bellow for matplotlib
plt.rcParams['axes.axisbelow'] = True

# Get Parser
parser = get_general_parser()
# Parse the argument
args = parser.parse_args()
args_dict = vars(args)


# Set manual seeds and get cuda
seed_everything(17)
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Handle the possibility of an all vs one binary problem
complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']
if args.all_vs_one=='False':
    binary_dict = {}
else:
    binary_dict = {label: 0 for label in complete_label_list}
    binary_dict[args.all_vs_one] = 1

# Obtain dataset depending on the args specified source
dataset = get_dataset_from_args(args)

# Declare all the saving paths needed
path_dict = get_paths(args.exp_name)

# Create results directory
if not os.path.isdir(path_dict['results']):
    os.makedirs(path_dict['results'])

fold_performance = {}

# Get expression data and metadata
expression_data = dataset.gene_filtered_data_matrix.T
metadata = dataset.categories_filtered
folds = dataset.k_fold_indexes

### Add fold information to the metadata
# Start with a big fold matrix
fold_matrix = -1*np.ones(len(metadata))
# Iterate over folds
for k, v in folds.items():
    # Get indexes of the test samples in the fold
    test_idx = v['test_index']
    # Add fold number to the fold matrix
    fold_matrix[test_idx] = k
# Add fold matrix to a metadata column
metadata['fold'] = fold_matrix.astype(int)
metadata.sort_index(inplace=True)

# Prediction dataframe
global_test_preds = pd.DataFrame(index=metadata.index, columns=dataset.lab_txt_2_lab_num.keys())

# Define fold performance dictionary
fold_performance = {}

# Iterate over folds
for i in range(args.fold_number):
    
    # Define the type of metod to use. Use default sklearn parameters for Decision Tree Classifier (dt),
    # Random Forest Classifier (rf), Extra Trees Classifier (et), Support Vector Machine (svm), Stochastic Gradient Descent (sgd),
    # K-Nearest Neighbors (knn).

    if args.sota == 'dt':
        clf =  DecisionTreeClassifier(random_state = 42)
    elif args.sota == 'rf':
        clf =  RandomForestClassifier(random_state = 42, n_jobs=-1)
    elif args.sota == 'et':
        clf =  ExtraTreesClassifier(random_state = 42, n_jobs=-1)
    elif args.sota == 'svm':
        clf =  SVC(kernel = 'rbf', random_state = 42, probability=True)
    elif args.sota == 'sgd':
        clf =  SGDClassifier(random_state = 42, n_jobs=-1, loss='log_loss')
    elif args.sota == 'knn':
        clf =  KNeighborsClassifier(n_jobs=-1)
    else:
        raise ValueError('The ml_method argument must be one of the following: dt, rf, et, svm, sgd, knn.')

    # Get the train and test indexes of the fold
    train_idx = metadata[metadata['fold']!=i].index
    test_idx = metadata[metadata['fold']==i].index

    # Get the train and test data of the fold
    train_data = expression_data.loc[train_idx]
    test_data = expression_data.loc[test_idx]

    # Get the train and test labels of the fold
    train_labels = metadata.loc[train_idx]['lab_num'].values
    test_labels = metadata.loc[test_idx]['lab_num'].values

    # Fit the model
    clf.fit(train_data.values, train_labels)

    # Get the predictions of the fold
    test_preds = clf.predict_proba(test_data.values)

    # Get predictions in a dataframe
    test_preds = pd.DataFrame(test_preds, index=test_idx, columns=dataset.lab_txt_2_lab_num.keys())
    
    # Accumulate the predictions of the fold in a global dataframe
    global_test_preds.loc[test_idx] = test_preds

    # Obtain test metrics for each epoch in all groups    
    test_metrics = get_metrics(test_preds, test_labels)
    fold_performance[i] = test_metrics

# Declare the invalid metrics for not considering them
invalid_metrics = ['conf_matrix', 'AP_list', 'pr_curve', 'correct_prob_df', 'pr_df']
# Get the scalar metrics for each fold
scalar_metrics = {fold: {k: v for k, v in fold_dict.items() if k not in invalid_metrics} for fold, fold_dict in fold_performance.items()}

# Create a dataframe with the scalar metrics
scalar_metrics_df = pd.DataFrame.from_dict(scalar_metrics, orient='index')
# Compute mean and std for each metric and add them to the bottom of the dataframe
scalar_metrics_df.loc['Mean'] = scalar_metrics_df.mean()
scalar_metrics_df.loc['Std'] = scalar_metrics_df.std()

# Save performance df to csv
scalar_metrics_df.to_csv(path_dict['metrics'])

# Define final print strings
macc_str = f'{round(100*scalar_metrics_df["mean_acc"].loc["Mean"], 1)} ± {round(100*scalar_metrics_df["mean_acc"].loc["Std"], 1)}'
tot_acc_str = f'{round(100*scalar_metrics_df["tot_acc"].loc["Mean"], 1)} ± {round(100*scalar_metrics_df["tot_acc"].loc["Std"], 1)}'
mean_AP_str = f'{round(100*scalar_metrics_df["mean_AP"].loc["Mean"], 1)} ± {round(100*scalar_metrics_df["mean_AP"].loc["Std"], 1)}'

# Open log file and print
with open(path_dict['train_log'], 'a') as f:
    print_both('\n', f)
    print_both(f'General results:', f)
    print_both(scalar_metrics_df, f)
    print_both('\n', f)
    print_both(f'Final performance = {macc_str}, {tot_acc_str}, {mean_AP_str}', f)

# Save the predictions of the best model and the ground truths
global_test_preds.to_csv(path_dict['predictions'])
metadata.to_csv(path_dict['groundtruths'])
