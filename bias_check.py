from thundersvm import SVC
from sklearn.metrics import classification_report
import os
import pylab
import time
# Import auxiliary functions
from utils import *
from datasets import *
from batch_metrics import *


# Set axis bellow for matplotlib
plt.rcParams['axes.axisbelow'] = True
# Set figure fontsizes
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

# Get Parser
parser = get_general_parser()
# Parse the argument
args = parser.parse_args()
args_dict = vars(args)

# Print the classification output
# FIXME: All the paths should be centralized in the get_paths function from utils.py
bias_directory = os.path.join('results', 'bias_check')
os.makedirs(os.path.join(bias_directory, args.exp_name), exist_ok=True)

# Start timer
start = time.time()

# Obtain dataset depending on the args specified source
dataset = get_dataset_from_args(args)
# Set wang level to 0, and batch norm to false to obtain unprocessed data
prev_wang_level = args.wang_level
prev_batch_norm = args.batch_norm
args.wang_level = 0
args.batch_norm = 'None'
unprocessed_dataset = get_dataset_from_args(args)
# Restore wang level
args.wang_level = prev_wang_level
args.batch_norm = prev_batch_norm

# Get adata from datasets
adata = get_adata_from_dataset(dataset)
unprocessed_adata = get_adata_from_dataset(unprocessed_dataset)

# Process adatas to compute batch metrics
adata = process_adata(adata)
unprocessed_adata = process_adata(unprocessed_adata)

# Get batch correction metrics
batch_metrics_dict = get_batch_correction_metrics(adata, unprocessed_adata)
# Get biological metrics
biological_metrics_dict = get_biological_conservation_metrics(adata, unprocessed_adata)

# Join both dictionaries
metrics_dict = {**batch_metrics_dict, **biological_metrics_dict}

# Compute global score and add it to the dictionary
metrics_dict['GLOBAL_SCORE'] = 0.6*metrics_dict['BIOLOGICAL_MEAN'] + 0.4*metrics_dict['CORRECTION_MEAN']

with open(os.path.join(bias_directory, args.exp_name, f'bias_log.txt'), 'a') as f:
        print_both('\n'.join(['--{0} {1}'.format(arg, args_dict[arg]) for arg in args_dict]),f)
        print_both('\n\n',f)
        print_both(f'\nTotal time to get integration metrics: {time.time() - start:.2f} seconds',f)
        print_both('\n\n',f)
        print_both('\n'.join(['{0}: {1}'.format(met, metrics_dict[met]) for met in metrics_dict]), f)


# Get a split of the zero fold
split_dict = dataset.get_batch_split(fold=0)

# Declare and fit linear Support Vector Machine
# TODO: Apply 5 fold cross validation
clf = SVC(kernel='linear', degree=1, verbose=True, max_iter=-1, tol=1e-4)
print('The linear SVM fit may take several minutes...')
clf.fit(split_dict['x']['train'].T, split_dict['y']['train']) 

# Get predictions
y_pred = clf.predict(split_dict['x']['test'].T)

with open(os.path.join(bias_directory, args.exp_name, f'bias_log.txt'), 'a') as f:
        print_both('\n'.join(['--{0} {1}'.format(arg, args_dict[arg]) for arg in args_dict]),f)
        print_both('\n\n',f)
        print_both(classification_report(split_dict['y']['test'], y_pred), f)


# This is the old code with various valuable plots
# ######################################################################
# #            You can safely change these parameters                  #
# ######################################################################
# dataset_to_check = 'toil_norm' # toil , wang or toil_norm
# sample_frac = 0.5 # Float [0,1) minimum fraction of samples in which each gene is expressed
# ######################################################################

# # Make directory to save bias separation histograms
# if not os.path.exists(os.path.join('Figures')):
#     os.makedirs(os.path.join('Figures'))


# if (dataset_to_check=='toil') or (dataset_to_check=='toil_norm'):

#     # Define list of complete labels
#     complete_label_list = [ 'GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER',
#                             'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS',
#                             'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT',
#                             'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA',
#                             'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH',
#                             'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO',
#                             'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD',
#                             'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']

#     # Define binary dict with the TCGA data labeled as 1
#     binary_dict = {}
#     for label in complete_label_list:
#         if label[:4] == 'GTEX':
#             binary_dict[label] = 0
#         else:
#             binary_dict[label] = 1

#     # Define normalization
#     norm_str = 'None' if dataset_to_check=='toil' else 'normal'

#     # Declare dataset
#     dataset = ToilDataset(os.path.join("data", "toil_data"),
#                                 dataset = 'both',
#                                 tissue = 'all',
#                                 binary_dict=binary_dict,
#                                 mean_thr = -10.0,
#                                 std_thr = 0.0,
#                                 rand_frac=1.0,
#                                 sample_frac=sample_frac,
#                                 label_type = 'phenotype',
#                                 batch_normalization=norm_str,
#                                 partition_seed = 0,
#                                 force_compute = False)


#     labels = dataset.split_labels
#     data = dataset.split_matrices

#     # Get binary labels: 1 == sample is from TCGA
#     x_train = data['train'].T
#     y_train = labels['train'].index.str.contains('TCGA')

#     x_val = data['val'].T
#     y_val = labels['val'].index.str.contains('TCGA')

#     if (dataset_to_check=='toil_norm') & (sample_frac==0.0):
#         # Make Figure
#         fig, axes = plt.subplots(1, 2, figsize=(15,7))
#         x_train.loc[y_train, :].plot(x='ENSG00000251953.1', y='ENSG00000278813.1', kind='scatter', c='darkcyan', ax=axes[0])
#         x_train.loc[~y_train, :].plot(x='ENSG00000251953.1', y='ENSG00000278813.1', kind='scatter', c = 'k', ax=axes[0])
#         axes[0].set_title('Scatter of 2 Genes Expressed in\n Less Than 0.2% of Samples')
#         axes[0].spines['top'].set_visible(False)
#         axes[0].spines['right'].set_visible(False)
#         axins = axes[0].inset_axes([0.3, 0.3, 0.47, 0.47])
#         x_train.loc[y_train, :].plot(x='ENSG00000251953.1', y='ENSG00000278813.1', kind='scatter', xlim=(-0.1,0.1), ylim = (-0.1,0.1), c='darkcyan', ax=axins, xlabel='', ylabel='')
#         x_train.loc[~y_train, :].plot(x='ENSG00000251953.1', y='ENSG00000278813.1', kind='scatter', xlim=(-0.1,0.1), ylim = (-0.1,0.1), c = 'k', ax=axins, xlabel='', ylabel='')
#         axins.set_title('99.6% of Samples')
#         axes[0].indicate_inset_zoom(axins, edgecolor="black")
#         axes[0].legend(['TCGA', 'GTEx'])
#         axes[0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=axes[0].transAxes, size=20, weight='bold')

#         # Get genes with sample frac between 0.5 and 0.51
#         valid_genes = dataset.general_stats[(dataset.general_stats['min_sample_frac']<0.51) & (dataset.general_stats['min_sample_frac']>0.50)].index.values
#         # Make figure
#         x_train.loc[y_train, :].plot(x=valid_genes[3], y=valid_genes[4], kind='scatter', c='darkcyan', ax = axes[1])
#         x_train.loc[~y_train, :].plot(x=valid_genes[3], y=valid_genes[4], kind='scatter', c = 'k', ax=axes[1])
#         axes[1].set_title('Scatter of 2 Genes Expressed in\nHalf the Samples')
#         axes[1].spines['top'].set_visible(False)
#         axes[1].spines['right'].set_visible(False)
#         axes[1].legend(['TCGA', 'GTEx'])
#         axes[1].text(-0.1, 1.1, string.ascii_uppercase[1], transform=axes[1].transAxes, size=20, weight='bold')
#         plt.tight_layout()
#         plt.savefig(os.path.join('Figures', 'centered_bias.png'), dpi=300)
#         plt.close()


#         # Make pca
#         pca = PCA(n_components=3)
#         pca.fit(x_train.T)
#         plt.figure()
#         plt.plot(pca.components_[:,y_train][1,:], pca.components_[:,y_train][2,:], '.', c='darkcyan')
#         plt.plot(pca.components_[:,~y_train][1,:], pca.components_[:,~y_train][2,:], '.', c='k')
#         plt.title('PCA of Normalized Toil Dataset')
#         plt.xlabel('Principal Component 2')
#         plt.ylabel('Principal Component 3')
#         plt.legend(['TCGA', 'GTEx'])
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         plt.tight_layout()
#         plt.savefig(os.path.join('Figures', 'pca_norm_toil_sample_frac_0.5.png'), dpi=300)
#         plt.close()
    
#     if (dataset_to_check=='toil'):
#         # Make pca
#         pca = PCA(n_components=3)
#         pca.fit(x_train.T)
#         plt.figure()
#         plt.plot(pca.components_[:,y_train][1,:], pca.components_[:,y_train][2,:], '.', c='darkcyan')
#         plt.plot(pca.components_[:,~y_train][1,:], pca.components_[:,~y_train][2,:], '.', c='k')
#         plt.title('PCA of Original Toil Dataset')
#         plt.xlabel('Principal Component 2')
#         plt.ylabel('Principal Component 3')
#         plt.legend(['TCGA', 'GTEx'])
#         ax = plt.gca()
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         plt.tight_layout()
#         plt.savefig(os.path.join('Figures', 'pca_toil.png'), dpi=300)
#         plt.close()

#     # Assign classifier depending on the dataset
#     if dataset_to_check=='toil':
#         clf = LinearSVC(random_state=0, verbose=2, max_iter=200000)
#     else:
#         # clf = SGDClassifier(max_iter=1000, verbose=2, n_jobs=-1, random_state=0, validation_fraction=0.2)
#         clf = LinearSVC(random_state=0, verbose=2, max_iter=200000)

#     clf.fit(x_train, y_train)

#     # Get predictions
#     y_pred = clf.predict(x_val)
#     print(classification_report(y_val, y_pred))

#     # Define save path for histogram
#     save_path = 'toil_svm_distance.png' if dataset_to_check=='toil' else 'normalized_toil_svm_distance.png'
#     save_path = os.path.join('Figures', save_path)

# elif dataset_to_check=='wang':

#     dataset = WangDataset(os.path.join('data', 'wang_data'))
#     train_index, val_index = train_test_split(dataset.categories["is_tcga"], test_size = 0.2, random_state = 0, stratify = dataset.categories["is_tcga"].values)
#     complete_dataset = dataset.matrix_data
#     del complete_dataset['Hugo_Symbol']
#     del complete_dataset['Gene_Hugo_Symbol']
#     x_train, x_val = complete_dataset.iloc[2:, train_index.index], complete_dataset.iloc[2:, val_index.index]
#     y_train = dataset.categories.loc[train_index.index, ['sample', 'is_tcga']].set_index('sample')
#     y_val = dataset.categories.loc[val_index.index, ['sample', 'is_tcga']].set_index('sample')

#     x_train = x_train.T.values
#     x_val = x_val.T.values

#     # Transforms
#     x_train = np.log2(x_train+1)
#     x_val = np.log2(x_val+1)

#     y_train = np.ravel(y_train.values)
#     y_val = np.ravel(y_val.values)

#     clf = LinearSVC(random_state=0, verbose=2, max_iter=200000)
#     clf.fit(x_train, y_train)

#     # Get predictions
#     y_pred = clf.predict(x_val)
#     print(classification_report(y_val, y_pred))

#     # Define save path for histogram
#     save_path = 'wang_svm_distance.png'
#     save_path = os.path.join('Figures', save_path)


# # This code plots the separation histogram #############################################
# # Get the norm of the hyperplane
# plane_norm = np.linalg.norm(clf.coef_)
# # Get decision function values
# dec_function = clf.decision_function(x_val)
# # Get distances
# dist_plane = dec_function/plane_norm

# # Distance of TCGA and GTEx specifically
# tcga_dist = dist_plane[y_val]
# gtex_dist = dist_plane[~y_val]

# # Plot and save histogram of distances 
# plt.figure(figsize=(17,5))
# if dataset_to_check == 'toil_norm':
#     plt.hist(tcga_dist, bins=100, color='#4c8682', label='TCGA', alpha=0.8) # , range=(-0.004,0.01) #, range=(-10,10)
#     plt.hist(gtex_dist, bins=100, color='k', label='GTEx', alpha=0.8) # , range=(-0.004,0.01) # , range=(-10,10)
# else:
#     plt.hist(tcga_dist, bins=40, color='#4c8682', label='TCGA', alpha=0.8)
#     plt.hist(gtex_dist, bins=40, color='k', label='GTEx', alpha=0.8)
# plt.title('Separation Histogram', fontsize=40)
# plt.xlabel('Distance from SVM Plane', fontsize=30)
# plt.ylabel('Frequency', fontsize=30)
# plt.legend(loc=2, fontsize=20)
# ax = plt.gca()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tight_layout()
# plt.savefig(save_path, dpi=300)
# plt.close()

