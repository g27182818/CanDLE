# Import of needed packages
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.decomposition import PCA
import scipy
import time
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
parser.add_argument('--batch_norm',     type=str,   default="normal",         help="Normalization to perform in each subset of the dataset",                                                          choices=["none", "normal", "healthy_tcga"])
parser.add_argument('--mean_thr',       type=float, default=-10.0,          help="Mean threshold to filter out genes in initial toil data. Genes accepted have mean expression estrictly greater.")
parser.add_argument('--std_thr',        type=float, default=0.0,            help="Standard deviation threshold to filter out genes in initial toil data. Genes accepted have std estrictly greater. Is is set to 0.1 by default to make fair comparation with CanDLE")
parser.add_argument('--rand_frac',      type=float, default=1.0,            help="Select a random fraction of the genes that survive the mean and std filtering.")
parser.add_argument('--sample_frac',    type=float, default=0.0,            help="Filter out genes that are not expressed in at least this fraction of both the GTEx and TCGA data.")
parser.add_argument('--gene_list_csv',  type=str,   default='None',         help="Path to csv file with a subset of genes to train CanDLE. The gene list overwrites all other gene filterings. Example: Rankings/100_candle_thresholds/at_least_3_cancer_types.csv")
# Parse the argument
args = parser.parse_args()
#----------------------------------------------------------------------------------------------------------------------#


# Adaptation of code taken from Quinn et al repository DOI: 10.3389/fgene.2019.00599 ################################

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


# Handle the possibility of an all vs one binary problem
complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']
tcga_label_list = [lab for lab in complete_label_list if 'TCGA' in lab]

# Empty max f1 and AP arrays
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
                                mean_thr = args.mean_thr,
                                std_thr = args.std_thr,
                                rand_frac = args.rand_frac,
                                sample_frac=args.sample_frac,
                                gene_list_csv = args.gene_list_csv,
                                label_type = 'phenotype',
                                batch_normalization=args.batch_norm,
                                partition_seed = 0,
                                force_compute = False)


    keep_info = .1
    beta = .999

    # Get first version matrices
    x_train = dataset.split_matrices['train']
    y_train = dataset.split_labels['train']
    x_val, x_test = dataset.split_matrices['val'], dataset.split_matrices['test']
    y_val, y_test = dataset.split_labels['val'], dataset.split_labels['test']

    # Filter training matrix to get just positive samples
    x_train = dataset.split_matrices['train'].loc[:,(y_train==1).tolist()]
    y_train = dataset.split_labels['train'][(y_train==1).tolist()]

    x_train, x_val, x_test = x_train.T, x_val.T, x_test.T

    # Get scores from model
    anomal_score,threshold = get_score_threshold(x_train, x_val)

    # Get probabilities
    anomal_prob = (anomal_score-anomal_score.min())/(anomal_score.max()-anomal_score.min())
    anomal_prob =  1-anomal_prob

    # Get and print metrics
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_val, anomal_prob)
    max_f1 = np.nanmax(2 * (precision * recall) / (precision + recall))
    AP = sklearn.metrics.average_precision_score(y_val, anomal_prob)

    print('-'*89)
    print('Results for '+actual_label+' :')
    print('max F1: {:.3f} | AP: {:.3f}'.format(max_f1,AP))
    print('-'*89)

    max_f1_list = np.hstack((max_f1_list, max_f1)) if max_f1_list.shape else max_f1
    ap_list = np.hstack((ap_list, AP)) if ap_list.shape else AP


    # Make separation plots 
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
    # Make directory to save separation plots
    if not os.path.exists(os.path.join('Figures','sota_detection')):
        os.makedirs(os.path.join('Figures','sota_detection'))
    fig.savefig(os.path.join('Figures','sota_detection', actual_label + '.png'))
    plt.close(fig)


print('Global results :')
print('mean max F1: {:.3f} | mean AP: {:.3f}'.format(np.mean(max_f1_list),np.mean(ap_list)))



