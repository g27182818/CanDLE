# Import of needed packages
import numpy as np
import os
import torch
from tqdm import tqdm
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.decomposition import PCA
import scipy
import time
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
parser.add_argument('--dataset',        type=str,   default="both",         help="Dataset to use",                                                                                                  choices=["both", "tcga", "gtex"])
parser.add_argument('--tissue',         type=str,   default="all",          help="Tissue to use from data",                                                                                         choices=['all', 'Bladder', 'Blood', 'Brain', 'Breast', 'Cervix', 'Colon', 'Connective', 'Esophagus', 'Kidney', 'Liver', 'Lung', 'Not Paired', 'Ovary', 'Pancreas', 'Prostate', 'Skin', 'Stomach', 'Testis', 'Thyroid', 'Uterus'])
parser.add_argument('--batch_norm',     type=str,   default="none",         help="Normalization to perform in each subset of the dataset",                                                          choices=["none", "normal", "healthy_tcga"])
# Parse the argument
args = parser.parse_args()
#############################################################

# ------------------- Important variable parameters -------------------------------------------------------------------#
# Miscellaneous parameters --------------------------------------------------------------------------------------------#
torch.manual_seed(12345)            # Set torch manual seed                                                            #
device = torch.device("cuda")       # Set cuda device                                                                  #
# Dataset parameters --------------------------------------------------------------------------------------------------#
val_fraction = 0.2                  # Fraction of the data used for validation                                         #
coor_thr = 0.6                      # Spearman correlation threshold for declaring graph topology                      #
p_value_thr = 0.05                  # P-value Spearman correlation threshold for declaring graph topology              #
# Model parameters ----------------------------------------------------------------------------------------------------#
hidd = 8                            # Hidden channels parameter for baseline model                                     #
# Training parameters -------------------------------------------------------------------------------------------------#
metric = 'both'                     # Evaluation metric for experiment. Can be 'acc', 'mAP' or 'both'                  #
# ---------------------------------------------------------------------------------------------------------------------#

mean_thr = -10.0  
std_thr = 0.1   
use_graph = False

# Code taken from Quinn et all repository ################################

def get_residuals(data,U):
    I = np.identity(data.shape[1])
    z = data.dot(I - U.dot(U.T))
    residuals = np.power(z,2).sum(axis=1) 
    return residuals

def get_score_threshold(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_nom,X_test_nom = scaler.transform(X_train), scaler.transform(X_test)        
    pca = PCA()
    pca.fit(X_train_nom)
    
    S,U = pca.explained_variance_, pca.components_.T
    cs = S.cumsum()
    K = int(np.where(cs >= cs[-1] * keep_info)[0][0] + 1)
    U = U[:,:K]
    
    print('Getting Residuals...')
    start = time.time()
    residuals = get_residuals(X_test_nom, U)
    end = time.time()
    print('Residuals computed in {:.3f} seconds'.format(end-start))

    c_beta = scipy.stats.norm.ppf(1 - beta)
    theta1 = sum(S[K+1:])
    theta2 = sum(S[K+1:]**2)
    theta3 = sum(S[K+1:]**3)
    h0 = 1 - ((2*theta1*theta3)/(3*theta2*theta2))
    Qbeta = theta1 * (((c_beta*np.sqrt(2*theta2*h0*h0)/theta1) + 1 + ((theta2*h0*(h0-1))/(theta1*theta1)) )**(1/h0))
    
    return residuals, Qbeta


# Handle the posibility of an all vs one binary problem
complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']
tcga_label_list = [lab for lab in complete_label_list if 'TCGA' in lab]

# # Test code
# tcga_label_list = tcga_label_list[:3]

max_f1_list = np.array([])
ap_list = np.array([])

for actual_label in tcga_label_list:

    binary_dict = {label: 0 for label in complete_label_list}
    binary_dict[actual_label] = 1

    # Declare dataset
    dataset = ToilDataset(os.path.join("data", "toil_data"),
                                dataset = args.dataset,
                                tissue = args.tissue,
                                binary_dict=binary_dict,
                                mean_thr = mean_thr,
                                std_thr = std_thr,
                                use_graph = use_graph,
                                corr_thr = coor_thr,
                                p_thr = p_value_thr,
                                label_type = 'phenotype',
                                batch_normalization = args.batch_norm,
                                partition_seed = 0,
                                force_compute = False)


    keep_info = .1
    beta = .999

    # Get first version matrices
    x_train = dataset.split_matrices['train']
    y_train = dataset.split_labels['train']
    x_val, x_test = dataset.split_matrices['val'], dataset.split_matrices['test']
    y_val, y_test = dataset.split_labels['val'], dataset.split_labels['test']

    # Filter training matrix to get just possitive samples
    x_train = dataset.split_matrices['train'].loc[:,(y_train==1).tolist()]
    y_train = dataset.split_labels['train'][(y_train==1).tolist()]

    x_train, x_val, x_test = x_train.T, x_val.T, x_test.T

    anomal_score,threshold = get_score_threshold(x_train, x_val)

    anomal_prob = (anomal_score-anomal_score.min())/(anomal_score.max()-anomal_score.min())
    anomal_prob =  1-anomal_prob

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_val, anomal_prob)
    max_f1 = np.nanmax(2 * (precision * recall) / (precision + recall))
    AP = sklearn.metrics.average_precision_score(y_val, anomal_prob)

    print('-'*89)
    print('Results for '+actual_label+' :')
    print('max F1: {:.3f} | AP: {:.3f}'.format(max_f1,AP))
    print('-'*89)

    max_f1_list = np.hstack((max_f1_list, max_f1)) if max_f1_list.shape else max_f1
    ap_list = np.hstack((ap_list, AP)) if ap_list.shape else AP


    data_idx = np.arange(len(y_val))
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.scatter(data_idx, anomal_score,s=3,label='normal', color='blue')
    plt.scatter(data_idx[y_val==1], anomal_score[y_val==1], color='red',s=3,label='tumor')
    threshold_line = np.ones(len(y_val)) * threshold
    plt.plot(data_idx, threshold_line, color='green')
    plt.yscale('log')
    plt.xlabel('Data point',fontsize=18)
    plt.ylabel('Residual signal',fontsize=18)
    plt.title(actual_label,fontsize=20)
    plt.legend(fontsize=18)
    plt.show()  
    fig.tight_layout()
    fig.savefig('figs_detection/' + actual_label + '.png')
    plt.close(fig)


print('Global results :')
print('max F1: {:.3f} | AP: {:.3f}'.format(np.mean(max_f1_list),np.mean(ap_list)))
breakpoint()



