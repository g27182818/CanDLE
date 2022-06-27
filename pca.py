from xml.etree.ElementInclude import DEFAULT_MAX_INCLUSION_DEPTH
import numpy as np
import pandas as pd
from datasets import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# This function obtains a PCA dataframe for a ToilDataset
def toil_pca(dataset, tissue_identifiers):
    # Transfer data to DataFrame
    labels = dataset.split_labels
    data = dataset.split_matrices

    # Join train, val, and test dataframes
    complete_data = pd.concat([data['train'], data['val'], data['test']], axis=1)
    complete_labels = pd.concat([labels['train'], labels['val'], labels['test']], axis=0)

    # Obtain valid samples to plot
    valid_sample_list = [complete_labels==id for id in tissue_identifiers]

    # Filter valid samples
    for i, valid_df in enumerate(valid_sample_list):
        valid_samples = valid_df if i == 0 else valid_samples + valid_df 

    # Filter valid data using valid sample names
    valid_samples_df = pd.DataFrame(complete_labels[valid_samples])
    valid_samples_names = complete_labels[valid_samples].index.tolist()
    valid_data = complete_data[valid_samples_names]

    # Run PCA on toil
    pca = PCA(n_components=2, random_state=15)
    computed_pc = pca.fit_transform(valid_data.T)
    xdf = pd.DataFrame(data = computed_pc, columns = ['PC1', 'PC2'], index=valid_samples_df.index)
    df = pd.concat([xdf, valid_samples_df], axis=1)
    df['sample'] = df.index

    return df

tissue_identifiers = [2, 20, 27]
# Load dataset without normalization
dataset_toil = ToilDataset(os.path.join("data", "toil_data"),
                            dataset = "both",
                            tissue = "all",
                            mean_thr = -10,
                            std_thr = -1,
                            use_graph = False,
                            corr_thr = 0.6,
                            p_thr = 0.05,
                            label_type = 'phenotype',
                            batch_normalization='none',
                            partition_seed = 0,
                            force_compute = False)

dataset_toil_normal = ToilDataset(os.path.join("data", "toil_data"),
                                    dataset = "both",
                                    tissue = "all",
                                    mean_thr = -10,
                                    std_thr = -1,
                                    use_graph = False,
                                    corr_thr = 0.6,
                                    p_thr = 0.05,
                                    label_type = 'phenotype',
                                    batch_normalization='normal',
                                    partition_seed = 0,
                                    force_compute = False)

df_toil = toil_pca(dataset_toil, tissue_identifiers)
df_toil_normal = toil_pca(dataset_toil_normal, tissue_identifiers)


# Get Wang data
dataset = WangDataset(os.path.join('data', 'wang_data'))
complete_data = dataset.matrix_data
del complete_data['Hugo_Symbol']
del complete_data['Gene_Hugo_Symbol']
data_labels = dataset.categories

categories = ['GTEX-BLADDER', 'TCGA-BLCA', 'GTEX-PROSTATE', 'TCGA-PRAD', 'GTEX-THYROID', 'TCGA-THCA']

valid_indexes =  data_labels['lab_txt'].isin(categories)
valid_data_wang = complete_data[complete_data.columns[valid_indexes]]
valid_data_wang = valid_data_wang.apply(lambda x: np.log2(x+1)) 
valid_categories_wang = data_labels.loc[valid_indexes, :]
valid_categories_wang = valid_categories_wang.set_index('sample')


# Run PCA on wang
pca_1 = PCA(n_components=2, random_state=15)
computed_pc_1 = pca_1.fit_transform(valid_data_wang.T)
xdf_1 = pd.DataFrame(data = computed_pc_1, columns = ['PC1', 'PC2'], index=valid_categories_wang.index)
df_1 = pd.concat([xdf_1, valid_categories_wang['lab_txt']], axis=1)


# Plot PCA
fig = plt.figure(figsize = (24,8))

# Plot Toil
ax = fig.add_subplot(1,3,1) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Toil Healthy Tissue PCA', fontsize = 24)
targets = ['Bladder', 'Prostate', 'Thyroid']
colors = [plt.cm.magma(0.3), plt.cm.magma(0.5), plt.cm.magma(0.65)]

for i in range(len(tissue_identifiers)):
    indicesToKeep = df_toil['lab_num'] == tissue_identifiers[i]
    indices_gtex = df_toil['sample'].str.contains('GTEX')
    indices_tcga = ~df_toil['sample'].str.contains('GTEX')

    ax.scatter(df_toil.loc[indicesToKeep*indices_gtex, 'PC1']
        , df_toil.loc[indicesToKeep*indices_gtex, 'PC2']
        , color = colors[i]
        , s = 20, label = 'GTEx {}'.format(targets[i]))

    ax.scatter(df_toil.loc[indicesToKeep*indices_tcga, 'PC1']
        , df_toil.loc[indicesToKeep*indices_tcga, 'PC2']
        , color = colors[i]
        , s = 50, marker = "+", label = 'TCGA {}'.format(targets[i]))

ax.legend()
ax.grid()
ax.set_axisbelow(True)
plt.xlim([-300, 400])
plt.ylim([-300, 400])
plt.tight_layout()


# Plot Wang
ax = fig.add_subplot(1,3,2) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Wang Healthy Tissue PCA', fontsize = 24)
targets = ['GTEx Bladder', 'TCGA Bladder', 'GTEx Prostate', 'TCGA Prostate', 'GTEx Thyroid', 'TCGA Thyroid']
colors = [plt.cm.magma(0.3), plt.cm.magma(0.5), plt.cm.magma(0.65)]

for i in range(len(categories)):
    indicesToKeep = df_1['lab_txt'].str.contains(categories[i]) 

    act_size = 50 if i%2 else 20
    marker = '+' if i%2 else 'o'

    ax.scatter(df_1.loc[indicesToKeep, 'PC1']
        , df_1.loc[indicesToKeep, 'PC2']
        , color = colors[i//2]
        , s = act_size,  marker = marker, label = targets[i])

ax.legend()
ax.grid()
ax.set_axisbelow(True)
plt.tight_layout()


# Plot Toil normal normalized
ax = fig.add_subplot(1,3,3) 
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('Toil Normalized Healthy Tissue PCA', fontsize = 24)
targets = ['Bladder', 'Prostate', 'Thyroid']
colors = [plt.cm.magma(0.3), plt.cm.magma(0.5), plt.cm.magma(0.65)]

for i in range(len(tissue_identifiers)):
    indicesToKeep = df_toil_normal['lab_num'] == tissue_identifiers[i]
    indices_gtex = df_toil_normal['sample'].str.contains('GTEX')
    indices_tcga = ~df_toil_normal['sample'].str.contains('GTEX')

    ax.scatter(df_toil_normal.loc[indicesToKeep*indices_gtex, 'PC1']
        , df_toil_normal.loc[indicesToKeep*indices_gtex, 'PC2']
        , color = colors[i]
        , s = 20, label = 'GTEx {}'.format(targets[i]))

    ax.scatter(df_toil_normal.loc[indicesToKeep*indices_tcga, 'PC1']
        , df_toil_normal.loc[indicesToKeep*indices_tcga, 'PC2']
        , color = colors[i]
        , s = 50, marker = "+", label = 'TCGA {}'.format(targets[i]))

ax.legend()
ax.grid()
ax.set_axisbelow(True)
plt.xlim([-100, 200])
plt.ylim([-100, 100])
plt.tight_layout()
plt.show()
plt.savefig('test_pca.png', dpi=200)