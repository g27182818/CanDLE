# Import of needed packages
import numpy as np
import os
import torch
from tqdm import tqdm
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn
from sklearn.decomposition import PCA
# Import auxiliary functions
from utils import *
from model import *
from datasets import *

#---------------- Parser code -------------------------------------------------------------------------------------------#
# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--dataset',        type=str,   default="both",         help="Dataset to use",                                                                                                  choices=["both", "tcga", "gtex"])
parser.add_argument('--tissue',         type=str,   default="all",          help="Tissue to use from data",                                                                                         choices=['all', 'Bladder', 'Blood', 'Brain', 'Breast', 'Cervix', 'Colon', 'Connective', 'Esophagus', 'Kidney', 'Liver', 'Lung', 'Not Paired', 'Ovary', 'Pancreas', 'Prostate', 'Skin', 'Stomach', 'Testis', 'Thyroid', 'Uterus'])
parser.add_argument('--all_vs_one',     type=str,   default='False',        help="If False solves a multiclass problem, if other string solves a binary problem with this as the positive class.",  choices=['False', 'GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM'])
parser.add_argument('--batch_norm',     type=str,   default="none",         help="Normalization to perform in each subset of the dataset",                                                          choices=["none", "normal", "healthy_tcga"])
parser.add_argument('--mean_thr',       type=float, default=-10.0,          help="Mean threshold to filter out genes in initial toil data. Genes accepted have mean expression estrictly greater.")
parser.add_argument('--std_thr',        type=float, default=0.0,            help="Standard deviation threshold to filter out genes in initial toil data. Genes accepted have std estrictly greater.")
parser.add_argument('--epochs',         type=int,   default=500,            help="Number of epochs. Defaults to 500 in Hongs model.")
parser.add_argument('--pca',            type=str,   default='False',        help="Wheter to perform PCA reduction to 2000 features as in Hong.",                                                    choices=['True', 'False'])

# Parse the argument
args = parser.parse_args()
#----------------------------------------------------------------------------------------------------------------------#

# Miscellaneous parameters --------------------------------------------------------------------------------------------#
gpu = '0'                           # GPU to train                                                                     #
torch.manual_seed(12345)            # Set torch manual seed                                                            #
device = torch.device("cuda")       # Set cuda device                                                                  #
# ---------------------------------------------------------------------------------------------------------------------#

# Set GPU to run code
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# Handle the posibility of an all vs one binary problem
complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']
if args.all_vs_one=='False':
    binary_dict = {}
else:
    binary_dict = {label: 0 for label in complete_label_list}
    binary_dict[args.all_vs_one] = 1

# Declare dataset
dataset = ToilDataset(os.path.join("data", "toil_data"),
                            dataset = args.dataset,
                            tissue = args.tissue,
                            binary_dict=binary_dict,
                            mean_thr = args.mean_thr,
                            std_thr = args.std_thr,
                            label_type = 'phenotype',
                            batch_normalization = args.batch_norm,
                            partition_seed = 0,
                            force_compute = False)

# Read lab_txt_2_tissue mapper
with open(os.path.join(dataset.path, "mappers", "id_2_tissue_mapper.json"), "r") as f:
    lab_txt_2_tissue = json.load(f)

# Complete the dictionary with classes that were previously not paired
lab_txt_2_tissue['GTEX-FAL_TUB'] = 'Fallopian tube'
lab_txt_2_tissue['GTEX-HEA'] = 'Heart'
lab_txt_2_tissue['GTEX-NER'] = 'Nerve'
lab_txt_2_tissue['GTEX-SAL_GLA'] = 'Salivary gland'
lab_txt_2_tissue['GTEX-SMA_INT'] = 'Small intestine'
lab_txt_2_tissue['GTEX-SPL'] = 'Spleen'
lab_txt_2_tissue['GTEX-VAG'] = 'Vagina'
lab_txt_2_tissue['TCGA-DLBC'] = 'Linphatic'
lab_txt_2_tissue['TCGA-HNSC'] = 'Mucosal Epithelium'
lab_txt_2_tissue['TCGA-THYM'] = 'Thymus'
lab_txt_2_tissue['TCGA-UVM'] = 'Uvea'

# get reversed tissue_2_lab_txt_tcga dictionary just for tcga labels
tissue_2_lab_txt_tcga = {}
for key, value in lab_txt_2_tissue.items():
    if 'TCGA' in key:
        tissue_2_lab_txt_tcga.setdefault(value, []).append(key)  

# Make dictionary that maps every lab_txt to a subtype
lab_txt_2_subtype = {}
for lab, tissue in lab_txt_2_tissue.items():
    if ('TCGA' in lab) and (len(tissue_2_lab_txt_tcga[tissue])>1):
        lab_txt_2_subtype[lab] = lab
    else:
        lab_txt_2_subtype[lab] = 'No Subtype'

# Compute binary cancer column in label_df
dataset.label_df['cancer'] = dataset.label_df['lab_txt'].str.contains('TCGA')

# Compute tissue of origin column in label_df
dataset.label_df['tissue'] = dataset.label_df['lab_txt'].map(lab_txt_2_tissue)

# compute subtype column in label_df
dataset.label_df['subtype'] = dataset.label_df['lab_txt'].map(lab_txt_2_subtype)

# Get numeric mappings of cancer, tissue and subtype annotations
cancer_2_num = {True:1, False:0}
tissue_2_num = {key:num for num, key in enumerate(sorted(dataset.label_df['tissue'].unique()))}
subtypes = sorted(list(lab_txt_2_subtype.values()))
subtypes = [subtype for subtype in subtypes if 'No Subtype' not in subtype]
subtype_2_num = {key:num for num, key in enumerate(subtypes)}
subtype_2_num['No Subtype'] = -1 # Important: -1 means that there is no subtype for that sample

# Get hong annotation list
hong_annot = list(zip(dataset.label_df['cancer'].map(cancer_2_num),
                        dataset.label_df['tissue'].map(tissue_2_num),
                        dataset.label_df['subtype'].map(subtype_2_num)))
dataset.label_df['hong_annot'] = hong_annot # Assign hong annotation column

# Get dictionaries from hong to toil annotations and viceversa
hong_2_toil_annot = {}
toil_2_hong_annot = {}
for lab in lab_txt_2_tissue.keys():
    cancer = 1 if 'TCGA' in lab else 0
    tissue = tissue_2_num[lab_txt_2_tissue[lab]]
    subtype = subtype_2_num[lab_txt_2_subtype[lab]]
    hong_2_toil_annot[(cancer, tissue, subtype)] = dataset.lab_txt_2_lab_num[lab]
    toil_2_hong_annot[dataset.lab_txt_2_lab_num[lab]] = (cancer, tissue, subtype)

# Obtain dataset annotations in hong format
hong_annotations = {key: pd.DataFrame(dataset.split_labels[key].map(toil_2_hong_annot)) for key in dataset.split_labels.keys()}
hong_annotations['train']['cancer'], hong_annotations['train']['tissue'], hong_annotations['train']['subtype'] = zip(*hong_annotations['train'].lab_num)
hong_annotations['val']['cancer'], hong_annotations['val']['tissue'], hong_annotations['val']['subtype'] = zip(*hong_annotations['val'].lab_num)
hong_annotations['test']['cancer'], hong_annotations['test']['tissue'], hong_annotations['test']['subtype'] = zip(*hong_annotations['test'].lab_num)

class HongDataset(Dataset):
    def __init__(self, matrix, annot, task='multitask'):
        self.task = task
        self.matrix = torch.Tensor(matrix.T.values).type(torch.float)
        self.annot = torch.Tensor(annot[['cancer', 'tissue', 'subtype']].values).type(torch.long)
        # Filter for subtype dataset
        if self.task == 'subtype':
            self.valid_indexes = self.annot[:, 2] != -1 # Find valid samples
            self.matrix = self.matrix[self.valid_indexes, :] # Modify matrix to just leave valid samples
            self.annot = self.annot[self.valid_indexes, :] # Modify annotations

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        sample = self.matrix[idx]
        label = self.annot[idx]
        return sample, label

# Handle the possible PCA in the input
if args.pca=='True':
    joint_matrix = pd.concat([dataset.split_matrices['train'], dataset.split_matrices['val'], dataset.split_matrices['test']], axis=1)
    pca = PCA(n_components=2000)
    print('Started computing PCA, this may take a few minutes...')
    pca.fit(joint_matrix)
    print('PCA computed. Working with the first 2000 components...')
    joint_matrix = pd.DataFrame(pca.components_, columns=joint_matrix.columns)
    dataset.split_matrices['train'] = joint_matrix[dataset.split_matrices['train'].columns]
    dataset.split_matrices['val'] = joint_matrix[dataset.split_matrices['val'].columns]
    dataset.split_matrices['test'] = joint_matrix[dataset.split_matrices['test'].columns]


# Declare datasets
# multitask
train_data = HongDataset(dataset.split_matrices['train'], hong_annotations['train'])
val_data = HongDataset(dataset.split_matrices['val'], hong_annotations['val'])
test_data = HongDataset(dataset.split_matrices['test'], hong_annotations['test'])
# subtask
subtype_train_data = HongDataset(dataset.split_matrices['train'], hong_annotations['train'], task='subtype')
subtype_val_data = HongDataset(dataset.split_matrices['val'], hong_annotations['val'], task='subtype')
subtype_test_data = HongDataset(dataset.split_matrices['test'], hong_annotations['test'], task='subtype')

# Declare dataloaders
# multitask
train_dataloader = DataLoader(train_data, batch_size=453, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=453, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=453, shuffle=False)
# subtask
subtype_train_dataloader = DataLoader(subtype_train_data, batch_size=421, shuffle=True)
subtype_val_dataloader = DataLoader(subtype_val_data, batch_size=421, shuffle=False)
subtype_test_dataloader = DataLoader(subtype_test_data, batch_size=421, shuffle=False)

# Handle input size in case we use a pca before
input_size = 2000 if args.pca=='True' else len(dataset.filtered_gene_list)

# Create models
hong_multitask_model = HongMultiTask(input_size = input_size).to(device)
hong_subtype_model = HongSubType(input_size = input_size).to(device)

# Declare criterions
cancer_criterion = torch.nn.CrossEntropyLoss()
tissue_criterion = torch.nn.CrossEntropyLoss()
subtype_criterion = torch.nn.CrossEntropyLoss()

# Handle differences in learning rates when training with PCA
if args.pca=='True':
    lr_multitask, lr_subtype = 6.3e-4, 4.9e-4
else:
    lr_multitask, lr_subtype = 6.3e-6, 4.9e-6

# Declare optimizers
multitask_optimizer = torch.optim.AdamW(hong_multitask_model.parameters(), lr=lr_multitask)
subtype_optimizer = torch.optim.AdamW(hong_subtype_model.parameters(), lr=lr_subtype)

# define train function multitask
def train_multitask(train_loader, model, device, cancer_criterion, tissue_criterion, optimizer):
    # Put model in train mode
    model.train()
    # Start the mean loss value
    mean_loss = 0
    mean_cancer_loss = 0
    mean_tissue_loss = 0
    # Start a counter
    count = 0
    with tqdm(train_loader, unit="batch") as t_train_loader:
        # Training cycle over the complete training batch
        for data in t_train_loader:  # Iterate in batches over the training dataset.
            t_train_loader.set_description(f"Batch {count+1}")
            input_cancer, input_tissue, input_subtype = data[1][:, 0].to(device), data[1][:, 1].to(device), data[1][:, 2].to(device)
            # Get the inputs of the model (x) and the groundtruth
            input_x = data[0].to(device)
            out_cancer, out_tissue = model(input_x)  # Perform a single forward pass.
            cancer_loss = cancer_criterion(out_cancer, input_cancer)  # Compute the cancer loss.
            tissue_loss = tissue_criterion(out_tissue, input_tissue)  # Compute the tissue loss.
            total_loss = cancer_loss + tissue_loss
            total_loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            mean_loss += total_loss
            mean_cancer_loss += cancer_loss
            mean_tissue_loss += tissue_loss
            count += 1
            # Update terminal descriptor
            t_train_loader.set_postfix(loss=mean_loss.item()/count)

    mean_loss = mean_loss/count
    mean_cancer_loss = mean_cancer_loss/count
    mean_tissue_loss = mean_tissue_loss/count
    return mean_loss, mean_cancer_loss, mean_tissue_loss

# define train function subtask
def train_subtask(train_loader, model, device, subtype_criterion, optimizer):
    # Put model in train mode
    model.train()
    # Start the mean loss value
    mean_loss = 0
    # Start a counter
    count = 0
    with tqdm(train_loader, unit="batch") as t_train_loader:
        # Training cycle over the complete training batch
        for data in t_train_loader:  # Iterate in batches over the training dataset.
            t_train_loader.set_description(f"Batch {count+1}")
            input_cancer, input_tissue, input_subtype = data[1][:, 0].to(device), data[1][:, 1].to(device), data[1][:, 2].to(device)
            # Get the inputs of the model (x) and the groundtruth
            input_x = data[0].to(device)
            out_subtype = model(input_x)  # Perform a single forward pass.
            subtype_loss = subtype_criterion(out_subtype, input_subtype)  # Compute the subtype loss.
            subtype_loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            mean_loss += subtype_loss
            count += 1
            # Update terminal descriptor
            t_train_loader.set_postfix(loss=mean_loss.item()/count)

    mean_loss = mean_loss/count
    return mean_loss


def test_multitask(loader, model, device):
    # Put model in evaluation mode
    model.eval()

    # Global true tensor
    glob_true_cancer = np.array([])
    glob_true_tissue = np.array([])
    # Global probability tensor
    glob_prob_cancer = np.array([])
    glob_prob_tissue = np.array([])

    count = 1
    # Computing loop
    with tqdm(loader, unit="batch") as t_loader:
        for data in t_loader:  # Iterate in batches over the training/test dataset.
            t_loader.set_description(f"Batch {count}")
            input_cancer, input_tissue, input_subtype = data[1][:, 0], data[1][:, 1], data[1][:, 2]
            # Get the inputs of the model (x) and the groundtruth
            input_x = data[0].to(device)
            out_cancer, out_tissue = model(input_x)  # Perform a single forward pass.
            # Get probabilities
            prob_cancer = out_cancer.softmax(dim=1).cpu().detach().numpy()
            prob_tissue = out_tissue.softmax(dim=1).cpu().detach().numpy()
            # Stack cases with previous ones
            glob_prob_cancer = np.vstack([glob_prob_cancer, prob_cancer]) if glob_prob_cancer.size else prob_cancer
            glob_prob_tissue = np.vstack([glob_prob_tissue, prob_tissue]) if glob_prob_tissue.size else prob_tissue
            glob_true_cancer = np.hstack((glob_true_cancer, input_cancer)) if glob_true_cancer.size else input_cancer
            glob_true_tissue = np.hstack((glob_true_tissue, input_tissue)) if glob_true_tissue.size else input_tissue
            # Update counter
            count += 1

    # Get predictions
    pred_cancer = glob_prob_cancer.argmax(axis=1)
    pred_tissue = glob_prob_tissue.argmax(axis=1)

    cancer_macc = sklearn.metrics.balanced_accuracy_score(glob_true_cancer, pred_cancer)
    tissue_macc = sklearn.metrics.balanced_accuracy_score(glob_true_tissue, pred_tissue)
    
    return cancer_macc, tissue_macc, pred_cancer, pred_tissue

def test_subtask(loader, model, device):
    # Put model in evaluation mode
    model.eval()

    # Global true tensor
    glob_true_subtype = np.array([])
    # Global probability tensor
    glob_prob_subtype = np.array([])

    count = 1
    # Computing loop
    with tqdm(loader, unit="batch") as t_loader:
        for data in t_loader:  # Iterate in batches over the training/test dataset.
            t_loader.set_description(f"Batch {count}")
            input_cancer, input_tissue, input_subtype = data[1][:, 0], data[1][:, 1], data[1][:, 2]
            # Get the inputs of the model (x) and the groundtruth
            input_x = data[0].to(device)
            out_subtype = model(input_x)  # Perform a single forward pass.
            # Get probabilities
            prob_subtype = out_subtype.softmax(dim=1).cpu().detach().numpy()
            # Stack cases with previous ones
            glob_prob_subtype = np.vstack([glob_prob_subtype, prob_subtype]) if glob_prob_subtype.size else prob_subtype
            glob_true_subtype = np.hstack((glob_true_subtype, input_subtype)) if glob_true_subtype.size else input_subtype
            # Update counter
            count += 1

    # Get predictions
    pred_subtype = glob_prob_subtype.argmax(axis=1)
    subtype_macc = sklearn.metrics.balanced_accuracy_score(glob_true_subtype, pred_subtype)
    
    return subtype_macc, pred_subtype

for i in range(args.epochs):
    # Train one epoch in multitask and subtype task
    total_loss, cancer_loss, tissue_loss = train_multitask(train_dataloader, hong_multitask_model, device,
                                                            cancer_criterion, tissue_criterion, multitask_optimizer)
    subtype_loss = train_subtask(subtype_train_dataloader, hong_subtype_model, device, subtype_criterion, subtype_optimizer)
    # Obtain metrics
    train_cancer_macc, train_tissue_macc, _, _ = test_multitask(train_dataloader, hong_multitask_model, device)
    val_cancer_macc, val_tissue_macc, _, _ = test_multitask(val_dataloader, hong_multitask_model, device)
    train_subtype_macc, _ = test_subtask(subtype_train_dataloader, hong_subtype_model, device)
    val_subtype_macc, _ = test_subtask(subtype_val_dataloader, hong_subtype_model, device)

    print('\nEpoch {} :\ntotal_loss = {:.2f}, cancer_loss = {:.2f}, tissue_loss = {:.2f}, subtype_loss = {:.2f}'.format(i, total_loss.cpu().detach().numpy(),
                                                                                                                            cancer_loss.cpu().detach().numpy(),
                                                                                                                            tissue_loss.cpu().detach().numpy(),
                                                                                                                            subtype_loss.cpu().detach().numpy()))
    print('Train: cancer macc: {:.3f} | tissue macc: {:.3f} | subtype macc: {:.3f}'.format(train_cancer_macc, train_tissue_macc, train_subtype_macc))
    print('Val  : cancer macc: {:.3f} | tissue macc: {:.3f} | subtype macc: {:.3f}\n'.format(val_cancer_macc, val_tissue_macc, val_subtype_macc))

_, _, val_cancer_pred, val_tissue_pred = test_multitask(val_dataloader, hong_multitask_model, device)
_, _, test_cancer_pred, test_tissue_pred = test_multitask(test_dataloader, hong_multitask_model, device)
_, val_subtype_pred = test_subtask(subtype_val_dataloader, hong_subtype_model, device)
_, test_subtype_pred = test_subtask(subtype_test_dataloader, hong_subtype_model, device)

# Handle subtype predictions to match the size of the ocmplete sets
val_complete_subtype = -1*np.ones_like(val_cancer_pred)
test_complete_subtype = -1*np.ones_like(test_cancer_pred)
val_complete_subtype[subtype_val_data.valid_indexes] = val_subtype_pred
test_complete_subtype[subtype_test_data.valid_indexes] = test_subtype_pred

# Join predictions
val_glob_pred = np.vstack([val_cancer_pred, val_tissue_pred, val_complete_subtype]).T
test_glob_pred = np.vstack([test_cancer_pred, test_tissue_pred, test_complete_subtype]).T

val_tuples = tuple(map(tuple, val_glob_pred))
test_tuples = tuple(map(tuple, test_glob_pred))

# Get toil format predictions from hong format predictions
# IMPORTANT: A -1 indicates that thoe hong model produced a prediction that is not vaid in the toil dataset
# This can be for example a cancer sample of lung tissue but from kidney kich subtype
val_toil_pred = [hong_2_toil_annot[tup] if tup in hong_2_toil_annot.keys() else -1 for tup in val_tuples]
test_toil_pred = [hong_2_toil_annot[tup] if tup in hong_2_toil_annot.keys() else -1 for tup in test_tuples]

val_toil_true = dataset.split_labels['val'].tolist()
test_toil_true = dataset.split_labels['test'].tolist()

macc_toil_val = sklearn.metrics.balanced_accuracy_score(val_toil_true, val_toil_pred)
acc_toil_val = sklearn.metrics.accuracy_score(val_toil_true, val_toil_pred)
macc_toil_test = sklearn.metrics.balanced_accuracy_score(test_toil_true, test_toil_pred)
acc_toil_test = sklearn.metrics.accuracy_score(test_toil_true, test_toil_pred)

print('Val : macc: {:.3f} | acc: {:.3f}'.format(macc_toil_val, acc_toil_val))
print('Test: macc: {:.3f} | acc: {:.3f}'.format(macc_toil_test, acc_toil_test))

