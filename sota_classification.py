# Import of needed packages
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
# Import auxiliary functions
from utils import *
from model import *
from datasets import *

# Get Parser
parser = get_dataset_parser()

# Add arguments for model training
parser.add_argument('--pca',            type=int,   default=-1,             help="Number of pca components to transform the data. If -1 then the original data is used.")
parser.add_argument('--exp_name',       type=str,   default='automatic',    help="Experiment name to save. If automatic is specified the name will be sota_classification/<<args.source>>/sample_frac_<<args.sample_frac>>_pca_<<args.pca>>")
parser.add_argument('--epochs',         type=int,   default=10,             help="Number of to train model.")
parser.add_argument('--gpu',            type=str,   default='0',            help="GPU to train model.")
# Parse the argument
args = parser.parse_args()


# Miscellaneous parameters --------------------------------------------------------------------------------------------#
torch.manual_seed(12345)            # Set torch manual seed                                                            #
device = torch.device("cuda")       # Set cuda device                                                                  #
# ---------------------------------------------------------------------------------------------------------------------#

# Set experiment name in case it is automatic
if args.exp_name == 'automatic':
    args.exp_name = os.path.join('sota_classification', f'{args.source}', f'sample_frac_{args.sample_frac}_pca_{args.pca}')
# Create directory for results
os.makedirs(os.path.join('results', args.exp_name), exist_ok=True)

# Set GPU to run code
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Handle the possibility of an all vs one binary problem
complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']
if args.all_vs_one=='False':
    binary_dict = {}
else:
    binary_dict = {label: 0 for label in complete_label_list}
    binary_dict[args.all_vs_one] = 1


# Declare dataset depending on the source
if args.source == 'toil':
    dataset = ToilDataset(  os.path.join('data', 'toil_data'),          dataset = args.dataset,
                            tissue = args.tissue,                       binary_dict=binary_dict,
                            mean_thr = args.mean_thr,                   std_thr = args.std_thr,
                            rand_frac = args.rand_frac,                 sample_frac=args.sample_frac,
                            gene_list_csv = args.gene_list_csv,         batch_normalization=args.batch_norm,
                            fold_number = args.fold_number,             partition_seed = args.seed,
                            force_compute = False)

elif args.source == 'wang':
    dataset = WangDataset(  os.path.join('data', 'wang_data'),          dataset = args.dataset,
                            tissue = args.tissue,                       binary_dict=binary_dict,
                            mean_thr = args.mean_thr,                   std_thr = args.std_thr,
                            rand_frac = args.rand_frac,                 sample_frac=args.sample_frac,
                            gene_list_csv = args.gene_list_csv,         batch_normalization=args.batch_norm,
                            fold_number = args.fold_number,             partition_seed = args.seed,
                            force_compute = False)

elif args.source == 'recount3':
    dataset = Recount3Dataset(os.path.join('data', 'recount3_data'),    dataset = args.dataset,
                            tissue = args.tissue,                       binary_dict=binary_dict,
                            mean_thr = args.mean_thr,                   std_thr = args.std_thr,
                            rand_frac = args.rand_frac,                 sample_frac=args.sample_frac,
                            gene_list_csv = args.gene_list_csv,         batch_normalization=args.batch_norm,
                            fold_number = args.fold_number,             partition_seed = args.seed,
                            force_compute = False)


# This dictionary of dataframes will store the global metric dataframe for each fold
fold_df_dict = {}

# Cycle over each fold
for fold in range(args.fold_number):

    # Get and declare data-loaders
    dataloader_dict = dataset.get_hong_dataloaders(batch_size_multitask=453, batch_size_subtype=421, pca=args.pca, fold=fold)
    train_multitask_dataloader, test_multitask_dataloader = dataloader_dict['multitask']
    train_subtype_dataloader, test_subtype_dataloader = dataloader_dict['subtype']
    
    # Get annotation dictionaries in both ways
    standard_2_hong_annot, hong_2_standard_annot = dataloader_dict['annot']

    # Handle input size in case we use a pca before
    input_size = len(dataset.filtered_gene_list) if args.pca == -1 else args.pca

    # Create models
    hong_multitask_model = HongMultiTask(input_size = input_size).to(device)
    hong_subtype_model = HongSubType(input_size = input_size).to(device)

    # Declare criterions
    cancer_criterion = torch.nn.CrossEntropyLoss()
    tissue_criterion = torch.nn.CrossEntropyLoss()
    subtype_criterion = torch.nn.CrossEntropyLoss()

    # Handle differences in learning rates when training with PCA
    if args.pca>0:
        lr_multitask, lr_subtype = 6.3e-4, 4.9e-4
    else:
        lr_multitask, lr_subtype = 6.3e-6, 4.9e-6

    # Declare optimizers
    multitask_optimizer = torch.optim.AdamW(hong_multitask_model.parameters(), lr=lr_multitask)
    subtype_optimizer = torch.optim.AdamW(hong_subtype_model.parameters(), lr=lr_subtype)

    # Declare fold dataframe
    fold_dataframe = pd.DataFrame(columns=[ 'tot_loss', 'can_loss', 'tis_loss', 'train_can_macc', 'train_tis_macc', 'train_sub_macc',
                                            'test_can_macc', 'test_tis_macc', 'test_sub_macc', 'test_standard_macc', 'test_standard_acc'])

    with tqdm(range(args.epochs), unit="epoch") as tqdm_epochs:
        tqdm_epochs.set_description(f"Fold {fold+1}")
        for i in tqdm_epochs:
            # Train one epoch in multitask and subtype task
            total_loss, cancer_loss, tissue_loss = train_hong_multitask(train_multitask_dataloader, hong_multitask_model, device, cancer_criterion, tissue_criterion, multitask_optimizer)
            subtype_loss = train_hong_subtask(train_subtype_dataloader, hong_subtype_model, device, subtype_criterion, subtype_optimizer)

            # Obtain metrics for multitask
            train_cancer_macc, train_tissue_macc, _, _ = test_hong_multitask(train_multitask_dataloader, hong_multitask_model, device)
            test_cancer_macc, test_tissue_macc, _, _ = test_hong_multitask(test_multitask_dataloader, hong_multitask_model, device)
            
            # Obtain metrics for subtype
            train_subtype_macc, _ = test_hong_subtask(train_subtype_dataloader, hong_subtype_model, device)
            test_subtype_macc, _ = test_hong_subtask(test_subtype_dataloader, hong_subtype_model, device)

            # Obtain metrics for test in standard format
            test_macc_standard, test_acc_standard = test_hong_with_standard_metrics(dataloader_dict, hong_multitask_model, hong_subtype_model, device)

            epoch_metrics = {   'tot_loss': tensor_2_numpy(total_loss),     'can_loss': tensor_2_numpy(cancer_loss),    'tis_loss': tensor_2_numpy(tissue_loss),
                                'train_can_macc': train_cancer_macc,        'train_tis_macc': train_tissue_macc,        'train_sub_macc': train_subtype_macc,
                                'test_can_macc': test_cancer_macc,          'test_tis_macc': test_tissue_macc,          'test_sub_macc': test_subtype_macc,
                                'test_standard_macc': test_macc_standard,   'test_standard_acc': test_acc_standard}

            # Save metrics in a global dataframe for each fold
            epoch_dataframe = pd.DataFrame(epoch_metrics, index=[i+1])
            fold_dataframe = pd.concat((fold_dataframe, epoch_dataframe))
            
            # If it is the fist datum it starts the log
            if i == 0:
                epoch_dataframe.to_csv(os.path.join('results', args.exp_name, f'log_fold_{fold}.csv'))
            # For the next data just append to the csv
            else:
                epoch_dataframe.iloc[[-1], :].to_csv(os.path.join('results', args.exp_name, f'log_fold_{fold}.csv'), mode='a', header=None)


            # Update terminal descriptor
            tqdm_epochs.set_postfix({'macc': test_macc_standard})

    fold_df_dict[fold] = fold_dataframe

all_folds_df = pd.concat(fold_df_dict, axis=1)
all_folds_df = all_folds_df.reorder_levels([1,0], axis=1)
final_metrics = all_folds_df[['test_standard_macc', 'test_standard_acc']].iloc[-1, :]
display_table = final_metrics.unstack(level=0)
descriptive_stats = display_table.describe()

final_tab = pd.concat([display_table, descriptive_stats.loc[['mean', 'std']]])
final_tab.to_csv(os.path.join('results', args.exp_name, 'final_metrics.csv'))
