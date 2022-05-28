from sklearn.ensemble import RandomForestClassifier
from datasets import *
from sklearn.metrics import balanced_accuracy_score

################ Temporal parser code #######################
################ Must be replace by configs #################
# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--adv_e_test',     type=float, default=0.01)
parser.add_argument('--adv_e_train',    type=float, default=0.00)
parser.add_argument('--n_iters_apgd',   type=int,   default=50)
parser.add_argument('--mode',           type=str,   default="test")
parser.add_argument('--num_test',       type=int,   default=69)
parser.add_argument('--train_samples',  type=int,   default=-1)
parser.add_argument('--dataset',        type=str,   default="both",         help="Dataset to use",                                      choices=["both", "tcga", "gtex"])
parser.add_argument('--tissue',         type=str,   default="all",          help="Tissue to use from data",                             choices=['all', 'Bladder', 'Blood', 'Brain', 'Breast', 'Cervix', 'Colon', 'Connective', 'Esophagus', 'Kidney', 'Liver', 'Lung', 'Not Paired', 'Ovary', 'Pancreas', 'Prostate', 'Skin', 'Stomach', 'Testis', 'Thyroid', 'Uterus'])
parser.add_argument('--model',          type=str,   default="MLP_ALL",      help="Model to use. Baseline is a graph neural network",    choices=["MLP_ALL", "MLP_FIL", "BASELINE"])
parser.add_argument('--lr',             type=float, default=0.00001,        help="Learning rate")
parser.add_argument('--batch_size',     type=int,   default=100,            help="Batch size")
parser.add_argument('--n_epochs',       type=int,   default=20,             help="Number of epochs")
parser.add_argument('--exp_name',       type=str,   default='misc_test',    help="EXperiment name to save")
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
dataset = args.dataset              # Dataset to use can be "both", "tcga" or "gtex"                                   #
tissue = args.tissue                # Tissue to use from data. "all" to use all tissues                                #
batch_size = args.batch_size        # Batch size parameter                                                             #
coor_thr = 0.6                      # Spearman correlation threshold for declaring graph topology                      #
p_value_thr = 0.05                  # P-value Spearman correlation threshold for declaring graph topology              #
# Model parameters ----------------------------------------------------------------------------------------------------#
hidd = 8                            # Hidden channels parameter for baseline model                                     #
model_type = args.model             # Model type, can be "MLP_FIL", "MLP_ALL", "BASELINE"                              #
# Training parameters -------------------------------------------------------------------------------------------------#
experiment_name = args.exp_name     # Experiment name to define path were results are stored                           #
lr = args.lr                        # Learning rate of the Adam optimizer (was changed from 0.001 to 0.00001)          #
total_epochs = args.n_epochs        # Total number of epochs to train                                                  #
metric = 'both'                     # Evaluation metric for experiment. Can be 'acc', 'mAP' or 'both'                  #
train_eps = args.adv_e_train        # Adversarial epsilon for train                                                    #
n_iters_apgd = args.n_iters_apgd    # Number of performed APGD iterations in train                                     #
# Test parameters -----------------------------------------------------------------------------------------------------#
test_eps = args.adv_e_test          # Adversarial epsilon for test                                                     #
# ---------------------------------------------------------------------------------------------------------------------#

# Handle input filtering depending in model type
if model_type == "MLP_FIL":
    mean_thr = 3.0  # Mean threshold for filtering input genes
    std_thr = 0.5   # Standard deviation threshold for filtering input genes
    use_graph = False
elif model_type == "BASELINE":
    mean_thr = 3.0  
    std_thr = 0.5   
    use_graph = True
elif model_type == "MLP_ALL":
    mean_thr = -10.0 
    std_thr = -1.0  
    use_graph = False
elif model_type == 'rf':
    mean_thr = -10.0
    std_thr = -1.0
    use_graph = False
else:
    raise NotImplementedError


dataset = ToilDataset(os.path.join("data", "toil_data"),
                            dataset = dataset,
                            tissue = tissue,
                            mean_thr = mean_thr,
                            std_thr = std_thr,
                            use_graph = use_graph,
                            corr_thr = coor_thr,
                            p_thr = p_value_thr,
                            label_type = 'phenotype',
                            partition_seed = 0,
                            force_compute = False)


labels = dataset.split_labels
data = dataset.split_matrices

x_train = data['train'].T
y_train = labels['train']


rf_model = RandomForestClassifier(n_estimators=100, verbose=2, random_state=0, n_jobs=-1)
rf_model.fit(x_train, y_train)

x_val = data['val'].T
y_val = labels['val']

# Get predictions
y_pred = rf_model.predict(x_val)

# Evaluate predictions
print(balanced_accuracy_score(y_val, y_pred))

importance_dict = dict(zip(dataset.filtered_gene_list, rf_model.feature_importances_))
# Print key value pairs from the top 20 higher values in dictionary
for key, value in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)[:20]:
    print(key, value)

breakpoint()