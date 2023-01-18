# Import of needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.decomposition import PCA
import scipy
import time
from tqdm import tqdm
# Import auxiliary functions
from utils import *
from model import *
from datasets import *

# Ignore a not necessary warning
np.seterr(divide='ignore', invalid='ignore')

# Get Parser
parser = get_dataset_parser()

# Add arguments for model training
parser.add_argument('--exp_name',       type=str,   default='automatic',    help="Experiment name to save. If automatic is specified the name will be sota_detection/<<args.source>>/sample_frac_<<args.sample_frac>>")
# Parse the argument
args = parser.parse_args()
args_dict = vars(args)

# Set experiment name in case it is automatic
if args.exp_name == 'automatic':
    args.exp_name = os.path.join('sota_detection', f'{args.source}', f'sample_frac_{args.sample_frac}')
# Create directory for results
os.makedirs(os.path.join('results', args.exp_name), exist_ok=True)

# Print experiment parameters
with open(os.path.join('results', args.exp_name, 'parser_logs.txt'), 'a') as f:
    print_both('Argument list to program',f)
    print_both('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                    for arg in args_dict]),f)
    print_both('\n\n',f)


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
    
    # print('Getting Residuals...')
    start = time.time()
    residuals = get_residuals(X_test_nom, U)
    end = time.time()
    # print('Residuals computed in {:.3f} seconds'.format(end-start))

    c_beta = scipy.stats.norm.ppf(1 - beta)
    theta1 = sum(S[K+1:])
    theta2 = sum(S[K+1:]**2)
    theta3 = sum(S[K+1:]**3)
    h0 = 1 - ((2*theta1*theta3)/(3*theta2*theta2))
    Qbeta = theta1 * (((c_beta*np.sqrt(2*theta2*h0*h0)/theta1) + 1 + ((theta2*h0*(h0-1))/(theta1*theta1)) )**(1/h0))
    
    return residuals, Qbeta

# Empty binary dict will be used because detection annotations will be handled from inside this code
binary_dict = {}

# A single dataset will be declared for each run and the binary detection annotations will be computed
# from inside the code. This is for saving computation time. 
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

# Obtain just TCGA labels
tcga_label_list = [lab for lab in dataset.lab_txt_2_lab_num.keys() if 'TCGA' in lab]
# Snippet to test the code with just two labels (comment in real use)
# tcga_label_list = tcga_label_list[:2]


# Declare a global metrics dataframe that will contain all
global_metrics_df = pd.DataFrame(columns = ['fold', 'label', 'max F1', 'AP'])


# Cycle over folds
for fold in tqdm(range(args.fold_number), position = 0, desc='Global progress'):
    # Get numpy split for the given fold
    split_dict = dataset.get_numpy_split(fold=fold)
    # Get original annotations
    train_orig_annot, test_orig_annot = split_dict['y']['train'], split_dict['y']['test']

    # Cycle over all the cancer labels
    for actual_label in tqdm(tcga_label_list, position = 1, leave=False, desc='Fold progress'):

        # Get the numeric label the we want to detect in this experiment
        num_label_to_detect = dataset.lab_txt_2_lab_num[actual_label]

        # Modify the original fold annotations to get binary annotations for the cancer type that we want to detect
        train_mod_annot = train_orig_annot == num_label_to_detect
        test_mod_annot = test_orig_annot == num_label_to_detect

        # Get first version data matrices
        x_train = split_dict['x']['train']
        x_test = split_dict['x']['test']
        
        # Filter training matrix to get just positive samples
        x_train = x_train.loc[:,(train_mod_annot==True).tolist()]

        # Transpose matrices to enter Quinn's method
        # The training matrix just contains positive samples while the test matrix contains both positive and negative samples
        x_train, x_test = x_train.T, x_test.T

        # Parameters specific to Quinn et al code
        keep_info = .1
        beta = .999

        # Get scores from model
        anomal_score,threshold = get_score_threshold(x_train, x_test)

        # Get probabilities (map scores to (0-1) according to the probability of not being a sample of the target cancer type)
        anomal_prob = (anomal_score-anomal_score.min())/(anomal_score.max()-anomal_score.min()) 
        # Invert probability to get the changes of being part of the desired cancer class
        anomal_prob =  1-anomal_prob

        # Get and print metrics
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(test_mod_annot, anomal_prob)
        max_f1 = np.nanmax(2 * (precision * recall) / (precision + recall))
        AP = sklearn.metrics.average_precision_score(test_mod_annot, anomal_prob)

        # Declare current metrics dataframe
        curr_df = pd.DataFrame({'fold': [fold], 'label': [actual_label], 'max F1': [max_f1], 'AP': [AP]})
        # Add curr_df to global metrics
        global_metrics_df = pd.concat([global_metrics_df, curr_df], ignore_index=True)

        # If it is the fist datum it starts the log
        if global_metrics_df.shape[0] == 1:
            global_metrics_df.to_csv(os.path.join('results', args.exp_name, f'metric_log.csv'))
        # For the next data just append to the csv
        else:
            global_metrics_df.iloc[[-1], :].to_csv(os.path.join('results', args.exp_name, f'metric_log.csv'), mode='a', header=None)


# Compute averages of all classes in each fold
mean_results = global_metrics_df.groupby('fold').mean(numeric_only=True)
# Obtain statistics that compare folds
results_stats = mean_results.describe()

# Print a final table with a summary of the results
final_table = pd.concat([mean_results, results_stats.loc[['mean', 'std']]])
final_table.columns = ['mean max F1', 'mAP']
print(final_table)

# Save to csv tha summary
final_table.to_csv(os.path.join('results', args.exp_name, 'final_metrics.csv'))