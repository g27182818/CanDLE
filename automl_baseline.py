# Import of needed packages
import numpy as np
import os
import torch
import h2o
from h2o.automl import H2OAutoML
import sys
from datetime import datetime
# Import auxiliary functions
from utils import *
from model import *
from datasets import *
from metrics import get_metrics
# Set matplotlib option to plot while in screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Start H2O API
h2o.init()

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
metadata['fold'] = fold_matrix

# Add to the expression data the fold of each sample
expression_data['fold'] = metadata['fold'].values
# Add to the expression data the numeric label of each sample
expression_data['lab_txt'] = metadata['lab_txt'].values

# Cast the lab_txt column to string for H2O to recognize it as categorical
expression_data['lab_txt'] = expression_data['lab_txt'].astype('str')

# Dataframe declaration
print('Parsing data to H2OFrame...')
expression_data_h2o = h2o.H2OFrame(expression_data)

# Declare name of dependent variable and fold column
y = 'lab_txt'
fold_column = 'fold'
# Get the names of the columns that are not the dependent variable or the fold column
x = expression_data_h2o.columns
x.remove(y)
x.remove(fold_column)

# Declare and perform algorithm training. As project name we use the date and time of the experiment
aml = H2OAutoML(
    max_runtime_secs = args.max_time,
    project_name = "experiment_"+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    keep_cross_validation_predictions=True,
    exclude_algos=['StackedEnsemble'],
    verbosity='info',
    seed = 42,
    )

# Train all models 
aml.train(x = x, y = y, fold_column=fold_column,  training_frame = expression_data_h2o)

# Get AutoML event log
log = aml.event_log
# Get training timing info
info = aml.training_info

# Get detailed performance of leader model
best_model = aml.leader
h2o.save_model(model=best_model, path=path_dict['results'], force=True)

# Get predictions of the best model in each fold
preds = best_model.cross_validation_predictions()

# Get the AutoML Leaderboard
lb = aml.leaderboard

# Define fold performance dictionary
fold_performance = {}

# Iterate over folds
for i in range(args.fold_number):
    
    # Get the predictions of the fold
    test_preds = preds[i].as_data_frame()
    # Drop the predict column from the predictions
    test_preds = test_preds.drop(columns=['predict'])
    # Accumulate the predictions of the fold in a global dataframe
    global_test_preds = test_preds if i==0 else global_test_preds.add(test_preds)

    # Get the indexes of the test samples of the fold
    test_idx = folds[i]['test_index']    
    # Subset the predictions to those only of the fold and get the ground truths
    test_preds_fold = test_preds.iloc[test_idx]
    test_truth_fold = metadata.iloc[test_idx]['lab_num'].values

    # Map the column names to the numeric labels
    test_preds_fold.columns = test_preds_fold.columns.map(dataset.lab_txt_2_lab_num)
    # Sort the columns by the numeric labels
    test_preds_fold = test_preds_fold.reindex(sorted(test_preds_fold.columns), axis=1)

    # Obtain test metrics for each epoch in all groups    
    test_metrics = get_metrics(test_preds_fold, test_truth_fold)
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
    print_both(f'AutoML training log:', f)
    print_both(log, f)
    print_both('\n', f)
    print_both(f'AutoML training info:', f)
    print_both(info, f)
    print_both('\n', f)
    print_both(f'Leaderboard of AuntoML (Ensembles excluded):', f)
    print_both(lb.head(rows=lb.nrows), f)
    print_both('\n', f)
    print_both(f'General results of AuntoML (Ensembles excluded):', f)
    print_both(scalar_metrics_df, f)
    print_both('\n', f)
    print_both(f'Final performance = {macc_str}, {tot_acc_str}, {mean_AP_str}', f)

# Save the predictions of the best model and the ground truths
global_test_preds.to_csv(path_dict['predictions'])
metadata.to_csv(path_dict['groundtruths'])

h2o.cluster().shutdown()