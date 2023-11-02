# Import of needed packages
import numpy as np
import os
import torch
from sklearn.ensemble import RandomForestClassifier
# Import auxiliary functions
from utils import *
from model import *
from datasets import *
from metrics import get_metrics
# Set matplotlib option to plot while in screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import optuna


# Set axis bellow for matplotlib
plt.rcParams['axes.axisbelow'] = True

# Get Parser
parser = get_general_parser()
# Parse the argument
args = parser.parse_args()
args_dict = vars(args)

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


def objective(trial):

    # Prediction dataframe
    global_test_preds = pd.DataFrame(index=metadata.index, columns=dataset.lab_txt_2_lab_num.keys())

    # Define fold performance dictionary
    fold_performance = {}

    # Iterate over folds
    for i in range(args.fold_number):
        
        # Define Random Forest Classifier
        clf =  RandomForestClassifier(
            n_estimators=trial.suggest_categorical('n_estimators', [128, 256, 512, 1024, 2048]),
            criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
            max_depth=None,
            min_samples_split = trial.suggest_categorical('min_samples_split', [2, 4, 8, 16, 32]),
            min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2, 4, 8]),
            min_weight_fraction_leaf=0.0,
            max_features=trial.suggest_categorical('max_features', [100, 200, 300]),
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=1)
        
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

    return scalar_metrics_df.loc['Mean', 'mean_acc']

study = optuna.create_study(study_name='rf_auto_ml',storage='sqlite:///db.sqlite3', direction='maximize')
study.optimize(objective, timeout=60*60*12)

print("Study statistics:")
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("Value:", trial.value)
print("Params:")
for key, value in trial.params.items():
    print("{}: {}".format(key, value))

# To open dashboard in host use: optuna-dashboard sqlite:///db.sqlite3





