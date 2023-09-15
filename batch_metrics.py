import anndata as ad
import pandas as pd
import scib
import scanpy as sc
import warnings
import numpy as np
import time

# FIlter intern numba umap warning that I could not fix
# TODO: Fix this warning
warnings.filterwarnings("ignore", message="The 'nopython' keyword argument was not supplied to the 'numba.jit' decorator. The implicit default value for this argument is currently False, but it will be changed to True in Numba 0.59.0. See https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-object-mode-fall-back-behaviour-when-using-jit for details.")

# Define mapping between classes and groups
group_2_lab = {
    1:  ['GTEX-PRO', 'TCGA-PRAD'],
    2:  ['GTEX-BLA', 'TCGA-BLCA'],
    3:  ['GTEX-BRE', 'TCGA-BRCA'],
    4:  ['GTEX-THY', 'TCGA-THCA'],
    5:  ['GTEX-STO', 'TCGA-STAD'],
    6:  ['GTEX-LUN', 'TCGA-LUAD', 'TCGA-LUSC'],
    7:  ['GTEX-LIV', 'TCGA-LIHC', 'TCGA-CHOL'],
    8:  ['GTEX-KID', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-KICH'],
    9:  ['GTEX-COL', 'TCGA-COAD', 'TCGA-READ'],
    10: ['GTEX-ESO', 'TCGA-ESCA'],
    11: ['GTEX-UTE', 'GTEX-CER', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-CESC'],
    12: ['GTEX-SAL_GLA', 'TCGA-HNSC']
}

# Get group tissues in text format
group_2_tissues = {
    1:  'Prostate',
    2:  'Bladder',
    3:  'Breast',
    4:  'Thyroid',
    5:  'Stomach',
    6:  'Lung',
    7:  'Liver',
    8:  'Kidney',
    9:  'Colon',
    10: 'Esophagus',
    11: 'Uterus',
    12: 'Head and Neck'
}

# Reverse dictionary
lab_2_group = {}
for group in group_2_lab:
    for lab in group_2_lab[group]:
        lab_2_group[lab] = group



def get_adata_from_dataset(dataset):
    """
    Get anndata object from dataset. This function also ensures that the samples in the adata object are
    only from the classes defined in group_2_lab. Finally it adds a column with the tissue number and
    another with the tissue name.

    Args:
        dataset (gtex_tcga_dataset): Dataset object. Can be wang, toil or recount3.

    Returns:
        ad.AnnData: Anndata object with the data from the dataset and the added columns.
    """
    # Get data
    X = dataset.gene_filtered_data_matrix.T
    obs = dataset.label_df

    # Assure the indexes are in the same order
    X.sort_index(inplace=True)
    obs.sort_index(inplace=True)

    adata = ad.AnnData(X=X, obs=obs)

    ### In case the dataset has not been sample filtered yet (processing level 0), we do it here
    # Get binary mask of samples with label in lab_2_group keys
    mask = adata.obs['lab_txt'].isin(lab_2_group.keys())

    # Filter adata
    adata = adata[mask, :].copy()

    # Add group column
    adata.obs['tissue'] = adata.obs['lab_txt'].map(lab_2_group)

    # Add group_txt column
    adata.obs['tissue_txt'] = adata.obs['tissue'].map(group_2_tissues)

    return adata

def process_adata(adata: ad.AnnData) -> ad.AnnData:
    """
    This function performs the preprocessing steps needed to compute the batch metrics. It performs PCA,
    finds nearest neighbors and computes the optimal clustering. All following the scib best practices that can
    be found in the original publication: https://doi.org/10.1038/s41592-021-01336-8

    Args:
        adata (ad.AnnData): Anndata object with the data from the dataset and the added columns.

    Returns:
        ad.AnnData: Processed anndata object with PCA, neighbors and optimal clustering computed.
    """
    print('Processing adata for batch metrics computation...')
    start = time.time()

    # Perform PCA and neighbors
    sc.pp.pca(adata, n_comps=50, copy=False)
    sc.pp.neighbors(adata, n_neighbors=15, copy=False)
    # Find optimal clustering
    scib.metrics.cluster_optimal_resolution(adata, label_key='tissue_txt', cluster_key='cluster', verbose=False)

    print(f'Done in {time.time() - start:.2f} seconds')
    
    return adata


def get_biological_conservation_metrics(adata: ad.AnnData, unprocessed_adata: ad.AnnData) -> dict:
    """
    This function takes a processed adata object and an unprocessed adata object and computes the biological signal 
    conservation metrics. The metrics computed are: ARI, NMI, ASW, SIL, CLISI, IL_F1 and HVG_OVERLAP. Details about
    each metric can be found in the scib documentation: https://scib.readthedocs.io/en/latest/api.html and in the
    original publication: https://doi.org/10.1038/s41592-021-01336-8

    Args:
        adata (ad.AnnData): Test anndata object already processed with process_adata function.
        unprocessed_adata (ad.AnnData): The initial anndata object withoun any batch integration. Must have been passed through process_adata function. 

    Returns:
        dict: Dictionary with the computed metrics.
    """

    # Cast tissue_txt and is_tcga to categorical
    adata.obs['tissue_txt'] = adata.obs['tissue_txt'].astype('category')
    unprocessed_adata.obs['tissue_txt'] = unprocessed_adata.obs['tissue_txt'].astype('category')

    adata.obs['is_tcga'] = adata.obs['is_tcga'].astype('category')
    unprocessed_adata.obs['is_tcga'] = unprocessed_adata.obs['is_tcga'].astype('category')

    # Start time tracking
    start = time.time()

    # Define metric dictionary
    metric_dict = {}

    # Compute metrics
    metric_dict['HVG_OVERLAP'] = scib.metrics.hvg_overlap(unprocessed_adata, adata, batch_key='is_tcga', n_hvg=500, verbose=False)
    metric_dict['ARI'] = scib.metrics.ari(adata, label_key='tissue_txt', cluster_key='cluster')
    metric_dict['NMI'] = scib.metrics.nmi(adata, label_key='tissue_txt', cluster_key='cluster')
    metric_dict['ASW'] = scib.metrics.isolated_labels_asw(adata, label_key="tissue_txt", batch_key='is_tcga', embed="X_pca", verbose=False, scale=True)
    metric_dict['SIL'] = scib.metrics.silhouette(adata, label_key='tissue_txt', embed="X_pca", scale=True)
    metric_dict['CLISI'] = scib.metrics.clisi_graph(adata, label_key='tissue_txt', type_='full', n_cores=-1, scale=True)
    metric_dict['IL_F1'] = scib.metrics.isolated_labels_f1(adata, label_key="tissue_txt", batch_key='is_tcga', embed=None, verbose=False)

    # Print time elapsed
    print(f'Computed biological metrics in {time.time() - start:.2f} seconds')

    # NOTE: The trajectory conservation metric is not computed here because it is not applicable to bulk RNASeq data.
    # NOTE: Cell cycle conservation is also not computed because it estimates a cell cycle phase for each sample and then computes the variance contribution
    #       of each cycle phase to the global batch or dataset. A cell cycle phase estimation should not be done over bulk RNS-Seq as it contain hundreds to thousands of cells.

    # Get list of the values of the metrics
    metric_values = list(metric_dict.values())
    # Add the mean of the values to the dictionary
    metric_dict['BIOLOGICAL_MEAN'] = np.mean(metric_values)

    return metric_dict


def get_batch_correction_metrics(adata: ad.AnnData, unprocessed_adata: ad.AnnData) -> dict:
    """
    This function takes a processed adata object and an unprocessed adata object and computes the batch correction
    metrics. The metrics computed are: GC, ILISI_GRAPH, KBET, PCR, SIL_BATCH. Details about each metric
    can be found in the scib documentation: https://scib.readthedocs.io/en/latest/api.html and in theoriginal
    publication: https://doi.org/10.1038/s41592-021-01336-8

    Args:
        adata (ad.AnnData): Test anndata object already processed with process_adata function.
        unprocessed_adata (ad.AnnData): The initial anndata object withoun any batch integration. Must have been passed through process_adata function. 

    Returns:
        dict: Dictionary with the computed metrics.
    """

    # Cast tissue_txt and is_tcga to categorical
    adata.obs['tissue_txt'] = adata.obs['tissue_txt'].astype('category')
    unprocessed_adata.obs['tissue_txt'] = unprocessed_adata.obs['tissue_txt'].astype('category')

    adata.obs['is_tcga'] = adata.obs['is_tcga'].astype('category')
    unprocessed_adata.obs['is_tcga'] = unprocessed_adata.obs['is_tcga'].astype('category')

    # Start time tracking
    start = time.time()
    
    # Define metric dictionary
    metric_dict = {}

    # Compute metrics
    metric_dict['GC'] = scib.metrics.graph_connectivity(adata, label_key='tissue_txt')
    metric_dict['ILISI_GRAPH'] = scib.metrics.ilisi_graph(adata, batch_key='is_tcga', type_='full', n_cores=-1)
    metric_dict['PCR'] = scib.metrics.pcr_comparison(unprocessed_adata, adata, covariate='is_tcga')
    metric_dict['SIL_BATCH'] = scib.metrics.silhouette_batch(adata, batch_key='is_tcga', label_key='tissue_txt', embed="X_pca", verbose=False)
    # metric_dict['KBET'] = scib.metrics.kBET(adata, batch_key='is_tcga', label_key='tissue_txt', type_="full", embed="X_pca")
    
    # Print time elapsed
    print(f'Computed batch correction metrics in {time.time() - start:.2f} seconds')

    # Get list of the values of the metrics
    metric_values = list(metric_dict.values())
    # Add the mean of the values to the dictionary
    metric_dict['CORRECTION_MEAN'] = np.mean(metric_values)

    return metric_dict
