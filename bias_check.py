from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import *
import os

# Set axis bellow for matplotlib
plt.rcParams['axes.axisbelow'] = True

######################################################################
#            You can safely change these parameters                  #
######################################################################
dataset_to_check = 'toil' # toil , wang or toil_norm
######################################################################

# Make directory to save bias separation histograms
if not os.path.exists(os.path.join('Figures')):
    os.makedirs(os.path.join('Figures'))


if (dataset_to_check=='toil') or (dataset_to_check=='toil_norm'):

    # Define list of complete labels
    complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']

    # Define binary dict with the TCGA data labeled as 1
    binary_dict = {}
    for label in complete_label_list:
        if label[:4] == 'GTEX':
            binary_dict[label] = 0
        else:
            binary_dict[label] = 1

    # Define normalization
    norm_str = 'none' if dataset_to_check=='toil' else 'normal'


    # Declare dataset
    dataset = ToilDataset(os.path.join("data", "toil_data"),
                                dataset = 'both',
                                tissue = 'all',
                                binary_dict=binary_dict,
                                mean_thr = -10.0,
                                std_thr = -0.1,
                                label_type = 'phenotype',
                                batch_normalization=norm_str,
                                partition_seed = 0,
                                force_compute = False)


    labels = dataset.split_labels
    data = dataset.split_matrices

    # Get binary labels: 1 == sample is from TCGA
    x_train = data['train'].T
    y_train = labels['train'].index.str.contains('TCGA')

    x_val = data['val'].T
    y_val = labels['val'].index.str.contains('TCGA')

    clf = LinearSVC(random_state=0, verbose=4, max_iter=100000)
    clf.fit(x_train, y_train)

    # Get predictions
    y_pred = clf.predict(x_val)
    print(classification_report(y_val, y_pred))

    # Define save path for histogram
    save_path = 'toil_svm_distance.png' if dataset_to_check=='toil' else 'normalized_toil_svm_distance.png'
    save_path = os.path.join('Figures', save_path)



elif dataset_to_check=='wang':

    dataset = WangDataset(os.path.join('data', 'wang_data'))
    train_index, val_index = train_test_split(dataset.categories["is_tcga"], test_size = 0.2, random_state = 0, stratify = dataset.categories["is_tcga"].values)
    complete_dataset = dataset.matrix_data
    del complete_dataset['Hugo_Symbol']
    del complete_dataset['Gene_Hugo_Symbol']
    x_train, x_val = complete_dataset.iloc[2:, train_index.index], complete_dataset.iloc[2:, val_index.index]
    y_train = dataset.categories.loc[train_index.index, ['sample', 'is_tcga']].set_index('sample')
    y_val = dataset.categories.loc[val_index.index, ['sample', 'is_tcga']].set_index('sample')

    x_train = x_train.T.values
    x_val = x_val.T.values

    # Transforms
    x_train = np.log2(x_train+1)
    x_val = np.log2(x_val+1)

    y_train = np.ravel(y_train.values)
    y_val = np.ravel(y_val.values)

    clf = LinearSVC(random_state=0, verbose=4, max_iter=100000)
    clf.fit(x_train, y_train)

    # Get predictions
    y_pred = clf.predict(x_val)
    print(classification_report(y_val, y_pred))

    # Define save path for histogram
    save_path = 'wang_svm_distance.png'
    save_path = os.path.join('Figures', save_path)


# This code plots the separation histogram #############################################
# Get the norm of the hiperplane
plane_norm = np.linalg.norm(clf.coef_)
# Get decision function values
dec_function = clf.decision_function(x_val)
# Get distances
dist_plane = dec_function/plane_norm

# Distance of TCGA and GTEx specifically
tcga_dist = dist_plane[y_val]
gtex_dist = dist_plane[~y_val]

# Plot and save histogram of distances 
plt.figure(figsize=(17,5))
if dataset_to_check == 'toil_norm':
    plt.hist(tcga_dist, bins=100, color='#4c8682', label='TCGA', alpha=0.8, range=(-10,10))
    plt.hist(gtex_dist, bins=100, color='k', label='GTEx', alpha=0.8, range=(-10,10))
else:
    plt.hist(tcga_dist, bins=40, color='#4c8682', label='TCGA', alpha=0.8)
    plt.hist(gtex_dist, bins=40, color='k', label='GTEx', alpha=0.8)
plt.title('Separation Histogram', fontsize=40)
plt.xlabel('Distance from SVM Plane', fontsize=30)
plt.ylabel('Frecuency', fontsize=30)
plt.legend(loc=2, fontsize=20)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
