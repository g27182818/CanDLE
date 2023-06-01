import subprocess
import tqdm
import pandas as pd
import numpy as np
import os
import time
import json
import pickle as pkl
import torch
import zipfile
import gzip
import shutil
import pylab
from rnanorm import TPM
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import anndata as ad
import scanpy as sc
import warnings
from typing import Callable, Tuple
from qnorm import quantile_normalize
from combat.pycombat import pycombat
from utils import *

# Suppress not useful warnings
pd.options.mode.chained_assignment = None  # default='warn'

# Set figure fontsizes
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

warnings.filterwarnings("ignore", category=UserWarning, module=r'.*rnanorm')

class gtex_tcga_dataset():
    def __init__(
            self,
            path:                   str,
            read_func:              Callable,
            dataset:                str = 'both',
            tissue:                 str = 'all',
            binary_dict:            dict = {},
            mean_thr:               float = -10.0,
            std_thr:                float = 0.01,
            rand_frac:              float = 1.0,
            sample_frac:            float = 0.5,
            gene_list_csv:          str = 'None',
            wang_level:             int = 0,
            batch_normalization:    str = 'None',
            fold_number:            int = 5,
            partition_seed:         int = 0,
            force_compute:          bool = False
            ):
        """
        This class is used to load the GTEx and TCGA datasets and perform the necessary preprocessing steps.
        It works with the 3 possible sources: Toil, Wang and Recount3.

        Args:
            path (str): Path where the raw data is stored.
            read_func (Callable): Function used to read the raw data. Different for each source.
            dataset (str, optional): Whether to use 'tcga', 'gtex' or 'both'. Defaults to 'both'.
            tissue (str, optional): Tissue to use from data. Note that the choices for source wang are limited by the available classes. Defaults to 'all'.
            binary_dict (dict, optional): Binary dict to map available labels to 0 or 1 numeric labels. This is used to make a binary detection problem. Defaults to {}.
            mean_thr (float, optional): Minimum mean expression for a gene to be considered. Defaults to -10.0.
            std_thr (float, optional): Minimum standard deviation needed for a gene to be considered. Defaults to 0.01.
            rand_frac (float, optional): Subset the samples to a random fraction between 0 and 1. Defaults to 1.0.
            sample_frac (float, optional): For a gene to be considered it should be expressed in at least this fraction of the samples in each batch (gtex, tcga). Defaults to 0.5.
            gene_list_csv (str, optional): Path to a gene csv that acts as a wildcard for gene filtering. Defaults to 'None'.
            wang_level (int, optional): Level of wang processing (0: Do not perform any wang processing, 1: Leave only paired samples, 2: Quantile normalization, 3: ComBat). Defaults to 0.
            batch_normalization (bool, optional): If true, performs z-score normalization in each batch separately. Defaults to True.
            fold_number (int, optional): Number of folds to divide the dataset. Defaults to 5.
            partition_seed (int, optional): Shuffle seed to perform k fold partition. Defaults to 0.
            force_compute (bool, optional): If true, recompute all statistics and re-do all pipeline from scratch. Defaults to False.
        """
        self.path = path
        self.read_func = read_func
        self.tissue = tissue
        self.binary_dict = binary_dict
        self.dataset = dataset
        self.dataset_info_path = os.path.join(self.path, 'processed_data',
                                              f'dataset={dataset}',
                                              f'mean_thr={mean_thr}_std_thr={std_thr}',
                                              f'sample_frac={sample_frac}_rand_frac={rand_frac}', 
                                              f'tissue={self.tissue}')
        self.mean_thr = mean_thr
        self.std_thr = std_thr
        self.rand_frac = rand_frac
        self.sample_frac = sample_frac
        self.gene_list_csv = gene_list_csv
        self.wang_level = wang_level
        self.batch_normalization = batch_normalization
        self.fold_number = fold_number
        self.partition_seed = partition_seed
        self.force_compute = force_compute

        # FIXME: Assert that wang dataset has a higher minimum wang_level

        # Main Bioinformatic pipeline
        # Make mapper files if they are not already saved
        self.make_mappers()
        
        # Read data from the Recount3 dataset and perform a log2(x+1) transformation. Also return gene metadata that is only available for Recount3
        self.matrix_data, self.categories = self.read_func(self.path, self.force_compute)
        
        # Make dataset directory if it does not exist
        os.makedirs(self.dataset_info_path, exist_ok = True)

        # Filter Wang datasets to use GTEx, TCGA or both.
        self.matrix_data_filtered, self.categories_filtered = self.filter_datasets()
        
        # Find stats of each dataset segment
        self.general_stats = self.find_general_stats()
        
        # Filter genes based on mean, std and sample_frac. This also subsamples the resulting filtered gene list by self.rand_frac. 
        # If self.gene_list_csv path is specified it works like a wildcard and CanDLE will train only with the genes in the csv path
        self.filtered_gene_list, self.gene_filtered_data_matrix = self.filter_genes()
        
        # Applies wang processing to self.gene_filtered_data_matrix
        self.gene_filtered_data_matrix, self.categories_filtered = self.wangify(self.gene_filtered_data_matrix, self.categories_filtered, self.wang_level)
        
        # Get labels dataframe and label dictionary. 
        self.label_df, self.lab_txt_2_lab_num = self.find_labels()
 
        # Perform batch normalization, this uses self.general_stats and normalizes self.gene_filtered_data_matrix  
        self.batch_normalize()
    
        # # Filter self.label_df and self.lab_txt_2_lab_num based on the specified tissue # TODO: Add filter by tissue function
        # self.filter_by_tissue()
        
        # Make the problem binary in case self.binary_dict is not empty
        self.make_binary_problem() # If self.binary_dict == {} this function does nothing
        
        # Get k fold cross validator. This is stratified
        self.k_fold_cross_validator = StratifiedKFold(n_splits=self.fold_number, shuffle=True, random_state=self.partition_seed)
        
        # Get k fold indexes
        self.k_fold_indexes = self.get_k_fold_indexes()
        
        # Define number of classes for classification
        self.num_classes = len(self.lab_txt_2_lab_num.keys()) if self.binary_dict == {} else 2
        
        # Define the number of samples
        self.num_samples = len(self.label_df)
        
        # Define the number of genes
        self.num_genes = len(self.filtered_gene_list)

        # Plot relevant plots here # TODO: Incorporate all the plots from Toil here
        # self.plot_dim_reduction()

    def make_mappers(self):
        """
        This function generates mapper files useful for class definition in the dataset by running the make_mappers.py file
        """
        # Just make mappers if they are not already saved
        if not os.path.exists(os.path.join(self.path, 'mappers')) or self.force_compute:
            # run main.py with subprocess
            command = f'python make_mappers.py'
            print(command)
            command = command.split()
            subprocess.call(command)

    # Filters the dataset by using or not using TCGA and GTEx samples.
    def filter_datasets(self):

        tcga_samples = self.categories.index[self.categories['is_tcga']]
        gtex_samples = self.categories.index[~self.categories['is_tcga']]

        # Handle the filters for TCGA and GTEx
        if self.dataset == 'tcga':
            print("Using TCGA samples only")
            # Filter out all gtex samples from matrix_data
            matrix_data_filtered = self.matrix_data.loc[:, tcga_samples]
            categories_filtered = self.categories.loc[tcga_samples, :]
        elif self.dataset == 'gtex':
            print("Using GTEx samples only")
            # Filter out all tcga samples from matrix_data
            matrix_data_filtered = self.matrix_data.loc[:, gtex_samples]
            categories_filtered = self.categories.loc[gtex_samples, :]
        elif self.dataset == 'both':
            # Do nothing because both TCGA and GTEX samples are included
            print("Using TCGA and GTEX samples")
            matrix_data_filtered = self.matrix_data
            categories_filtered = self.categories
            
        return matrix_data_filtered, categories_filtered

    # This function extracts the labels from categories and returns a label dataframe and a dictionary of textual labels to numeric labels
    def find_labels(self):

        # Initialize label df with filtered categories
        label_df = self.categories_filtered

        # Handle the only TCGA case where all normal samples have to be grouped in a single NT class
        if self.dataset == 'tcga':
            label_df.loc[label_df['lab_txt'].str.contains('GTEX'), 'lab_txt'] = 'TCGA-NT'

        # Get unique textual labels obtained and sort them
        current_labels = sorted(label_df["lab_txt"].unique().tolist())
        # Define lab_txt_2_lab_num dictionary
        lab_txt_2_lab_num = {lab_txt: i for i, lab_txt in enumerate(current_labels)}

        # Define numeric labels from the textual labels in label_df
        label_df["lab_num"] = label_df["lab_txt"].map(lab_txt_2_lab_num)
        
        # Save lab_txt_2_lab_num dictionary to json file
        with open(os.path.join(self.dataset_info_path, "lab_txt_2_lab_num_mapper.json"), "w") as f:
            json.dump(lab_txt_2_lab_num, f, indent = 4)

        return label_df, lab_txt_2_lab_num

    # TODO: Make that general stats is not hosted in self.path but in self.path/processed_data
    # This function finds the mean expression, std and expressed sample fraction for GTEx, TCGA, healthy TCGA and the joint dataset
    def find_general_stats(self):
        # If the info stats are already computed load them from file
        if (os.path.exists(os.path.join(self.path, 'general_stats.csv'))) & (self.force_compute == False):
            print('Loading general stats from '+os.path.join(self.path, 'general_stats.csv'))
            general_stats = pd.read_csv(os.path.join(self.path, 'general_stats.csv'), index_col = 0)
        # If the stats are not computed compute them and save them in file
        else:
            print('Computing general stats and saving to '+os.path.join(self.path, 'general_stats.csv'))
            # Define auxiliary tcga dataframe to obtain healthy tcga samples
            tcga_df = self.categories[self.categories['is_tcga']]

            # Get the identifiers of the samples in each subset
            gtex_samples = self.categories[self.categories['is_tcga']==False].index
            tcga_samples = tcga_df.index
            healthy_tcga_samples = tcga_df[tcga_df['lab_txt'].str.contains('GTEX')].index
            joint_samples = self.categories.index

            # Compute the mean of the subsets
            tqdm.pandas(desc="Computing Mean GTEx")
            gtex_mean = self.matrix_data.loc[:, gtex_samples].progress_apply(np.mean, axis = 1).to_frame(name='gtex_mean')
            tqdm.pandas(desc="Computing Mean TCGA")
            tcga_mean = self.matrix_data.loc[:, tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='tcga_mean')
            tqdm.pandas(desc="Computing Mean Healthy TCGA")
            healthy_tcga_mean = self.matrix_data.loc[:, healthy_tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='healthy_tcga_mean')
            tqdm.pandas(desc="Computing Joint Mean")
            joint_mean = self.matrix_data.loc[:, joint_samples].progress_apply(np.mean, axis = 1).to_frame(name='joint_mean')

            # Compute the std of the subsets
            tqdm.pandas(desc="Computing std GTEx")
            gtex_std = self.matrix_data.loc[:, gtex_samples].progress_apply(np.std, axis = 1).to_frame(name='gtex_std')
            tqdm.pandas(desc="Computing std TCGA")
            tcga_std = self.matrix_data.loc[:, tcga_samples].progress_apply(np.std, axis = 1).to_frame(name='tcga_std')
            tqdm.pandas(desc="Computing std Healthy TCGA")
            healthy_tcga_std = self.matrix_data.loc[:, healthy_tcga_samples].progress_apply(np.std, axis = 1).to_frame(name='healthy_tcga_std')
            tqdm.pandas(desc="Computing Joint std")
            joint_std = self.matrix_data.loc[:, joint_samples].progress_apply(np.std, axis = 1).to_frame(name='joint_std')

            # Compute the fraction of samples where a gene is expressed
            print('Computing fraction of samples where each gene is expressed ...')
            min_val = self.matrix_data.min().min() # Get minimum value
            tqdm.pandas(desc="Computing Expressed Genes")
            expressed_matrix = self.matrix_data.progress_apply(lambda x: x>min_val, axis = 1) # Compute expressed positions
            
            # Compute expressed sample fractions for all subsets
            tqdm.pandas(desc="Computing Joint Expressed Sample Fraction")
            joint_sample_fraction = expressed_matrix.progress_apply(np.mean, axis = 1).to_frame(name='joint_sample_frac')
            tqdm.pandas(desc="Computing GTEx Expressed Sample Fraction")
            gtex_sample_fraction = expressed_matrix.loc[:, gtex_samples].progress_apply(np.mean, axis = 1).to_frame(name='gtex_sample_frac')
            tqdm.pandas(desc="Computing TCGA Expressed Sample Fraction")
            tcga_sample_fraction = expressed_matrix.loc[:, tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='tcga_sample_frac')
            

            # Join stats in single dataframe
            general_stats = pd.concat([gtex_mean, tcga_mean, healthy_tcga_mean, joint_mean,
                                        gtex_std, tcga_std, healthy_tcga_std, joint_std,
                                        joint_sample_fraction, gtex_sample_fraction, tcga_sample_fraction,], axis=1)
            general_stats.to_csv(os.path.join(self.path, 'general_stats.csv'))

        return general_stats

    # This function filters out genes by mean, standard deviation, expression fraction, random fraction or list of genes
    def filter_genes(self):
        # If there is a gene list specified by parameter then it overwrites mean, std and rand_frac filtering  
        if self.gene_list_csv != 'None':
            # Print user message
            print(f'CanDLE will train with the list of genes specified in {self.gene_list_csv}')
            gene_csv_df = pd.read_csv(self.gene_list_csv, index_col=0)
            gene_list = pd.Index(gene_csv_df['gene_name'])
        
        # If no list of genes is specified then proceed with mean, std, sample_frac and rand_frac filtering
        elif (not os.path.exists(os.path.join(self.dataset_info_path, "filtering_info.csv"))) or self.force_compute:
            
            print("Computing list of filtered genes. And saving filtering info to:\n\t"+ os.path.join(self.dataset_info_path, "filtering_info.csv"))
            
            # Find the indices of the samples with mean, standard deviation and sample fractions that fulfill the thresholds
            mean_bool_index = ((self.general_stats['joint_mean']>self.mean_thr) & (self.general_stats['gtex_mean']>self.mean_thr) & (self.general_stats['tcga_mean']>self.mean_thr))
            std_bool_index = ((self.general_stats['joint_std']>self.std_thr) & (self.general_stats['gtex_std']>self.std_thr) & (self.general_stats['tcga_std']>self.std_thr))
            sample_frac_bool_index = ((self.general_stats['joint_sample_frac'] > self.sample_frac) & (self.general_stats['gtex_sample_frac'] > self.sample_frac) & (self.general_stats['tcga_sample_frac'] > self.sample_frac))
            
            # Compute intersection of mean_data_index and std_data_index
            mean_std_sample_index = np.logical_and.reduce((mean_bool_index.values, std_bool_index.values, sample_frac_bool_index)).ravel()
            # Make a gene list of the samples that fulfill the thresholds
            gene_list = self.matrix_data.index[mean_std_sample_index]

            # Subsample gene list in case self.rand_frac < 1
            if self.rand_frac < 1:
                np.random.seed(0) # Ensure reproducibility # TODO: Parametrize this seed to run variation experiments
                rand_selector = np.zeros(len(gene_list))
                rand_selector[:int(len(gene_list)*self.rand_frac)] = 1
                np.random.shuffle(rand_selector) # Shuffle boolean selector
                rand_selector = np.array(rand_selector, dtype=bool)
                gene_list = gene_list[rand_selector] # Filter gene list based in rand_selector
            
            # Compute boolean value for each gene that indicates if it was included in the filtered gene list
            included_in_filtering = self.general_stats.index.isin(gene_list)

            # Merge all statistics and included_in_filtering into a final dataframe
            filtering_info_df = self.general_stats
            filtering_info_df['included'] = included_in_filtering
            filtering_info_df.index.name = "gene"

            # Save the filtering info to files
            filtering_info_df.to_csv(os.path.join(self.dataset_info_path, "filtering_info.csv"), index = True)
            # Plot histograms with plot_filtering_histograms()
            self.plot_filtering_histograms(filtering_info_df)

        else:
            print("Loading filtering info from:\n\t" + os.path.join(self.dataset_info_path, "filtering_info.csv"))
            filtering_info_df = pd.read_csv(os.path.join(self.dataset_info_path, "filtering_info.csv"), index_col = 0)
            # get indices of filtering_info_df that are True in the included column
            gene_list = filtering_info_df.index[filtering_info_df["included"].values == True]
            # Plot histograms with plot_filtering_histograms()
            self.plot_filtering_histograms(filtering_info_df)
        
        # Filter tha data matrix based on the gene list
        gene_filtered_data_matrix = self.matrix_data_filtered.loc[gene_list, :]

        print("Currently working with {} genes...".format(gene_filtered_data_matrix.shape[0]))

        return gene_list.to_list(), gene_filtered_data_matrix

    def wangify(self, data_matrix: pd.DataFrame, category_df: pd.DataFrame, wang_level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function receives a data matrix and a category dataframe and returns a tuple of dataframes with the same
        format. However the protocols of the original wang paper (https://doi.org/10.1038/sdata.2018.61) are applied to the data matrix.
        Each part of the protocol is designated as a "wang_level" so the user inputs a level and the process is performed up to that level.
        The levels are:

            0:  Nothing is done to the data matrix. The data matrix is returned as is.
            1:  Samples from categories of the GTEx of TCGA that do not have sufficient paired samples in the other dataset are removed.
            2:  Reverse log2 transform, quantile normalize the data matrix and apply log2(x+1) transform.
            3:  ComBat batch correction is applied to each tissue type separately. In other words, for a given tissue type (e.g. Lung) we 
                perform ComBat batch correction on the samples of that tissue type from both datasets ('GTEX-LUN', 'TCGA-LUAD', 'TCGA-LUSC').
                The batch variable is the dataset (GTEx or TCGA) and the deceased state (healthy or tumor) is the biologically relevant
                variable (design matrix).


        Args:
            data_matrix (pd.DataFrame): Pandas dataframe with the data matrix. Rows are genes and columns are samples.
            category_df (pd.DataFrame): Pandas dataframe with the category information. Rows are samples and columns can vary
                                        but must contain at least the following columns: 'lab_txt', 'is_tcga'.
            wang_level (int):   Integer that indicates the level of wang preprocessing to apply. The levels are specified in the
                                function documentation.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tuple of dataframes with the same format as the input dataframes but with the
                                            wang preprocessing applied.
        """

        ### Define functions to perform each level of the wang preprocessing
        
        def wang_sample_filtering(data_matrix: pd.DataFrame, category_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            
            # Define categories that are used in the wang paper
            wang_categories = [ 'GTEX-BLA',     'GTEX-BRE',     'GTEX-CER',     'GTEX-COL',     'GTEX-ESO',
                                'GTEX-KID',     'GTEX-LIV',     'GTEX-LUN',     'GTEX-PRO',     'GTEX-SAL_GLA',
                                'GTEX-STO',     'GTEX-THY',     'GTEX-UTE',     'TCGA-BLCA',    'TCGA-BRCA',
                                'TCGA-CESC',    'TCGA-CHOL',    'TCGA-COAD',    'TCGA-ESCA',    'TCGA-HNSC',
                                'TCGA-KICH',    'TCGA-KIRC',    'TCGA-KIRP',    'TCGA-LIHC',    'TCGA-LUAD',
                                'TCGA-LUSC',    'TCGA-PRAD',    'TCGA-READ',    'TCGA-STAD',    'TCGA-THCA',
                                'TCGA-UCEC',    'TCGA-UCS']
            
            # Filter out samples that are not in the wang categories
            valid_samples = category_df.index[category_df['lab_txt'].isin(wang_categories)]

            # Filter out samples that are not in the wang categories
            valid_data_matrix = data_matrix[valid_samples]
            valid_category_df = category_df.loc[valid_samples]

            return valid_data_matrix, valid_category_df

        def wang_quantile_normalization(data_matrix: pd.DataFrame) -> pd.DataFrame:

            # Get the offset used in the log2 transform of the input data
            general_min = data_matrix.min().min()
            offset = np.power(2, general_min)

            print(f'Computed offset is {offset}')

            # Reverse log2 transform (using original values)
            orig_data_matrix = np.power(2, data_matrix) - offset

            # Quantile normalize data matrix
            qnorm_orig_data_matrix = quantile_normalize(orig_data_matrix, axis=1)

            # Do log2(x+1) transform
            qnorm_data_matrix = np.log2(qnorm_orig_data_matrix + 1)

            return qnorm_data_matrix

        def wang_combat(data_matrix: pd.DataFrame, category_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

            # Define wang groups to apply ComBat batch correction in each of them
            wang_groups = {
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

            # Define empty dictionary to store the batch corrected data
            batch_corrected_adata = {}
            
            # Cycle through wang groups and apply ComBat batch correction to each of them
            for group, categories in wang_groups.items():
                
                # Filter out samples that are not in the current wang group
                group_valid_samples = category_df.index[category_df['lab_txt'].isin(categories)]
                group_data_matrix = data_matrix[group_valid_samples]
                group_category_df = category_df.loc[group_valid_samples]

                # Create group adata object
                group_adata = ad.AnnData(group_data_matrix.T, obs=group_category_df)

                # Get sample decomposition for group
                healthy_tcga = (group_adata.obs['healthy'] & group_adata.obs['is_tcga']).sum()
                tumor_tcga = (~group_adata.obs['healthy'] & group_adata.obs['is_tcga']).sum()
                gtex_samples = (~group_adata.obs['is_tcga']).sum()
                print(f'group {group} with categories {categories} samples: GTEx {gtex_samples} | TCGA healthy {healthy_tcga} | TCGA tumor {tumor_tcga}') 

                # Apply ComBat batch correction to group adata
                try:
                    # Get expression matrix dataframe
                    group_df = group_adata.to_df().T
                    batch_list = group_adata.obs['is_tcga'].values.tolist()
                    healthy_list = group_adata.obs['healthy'].values.tolist()

                    # Apply pycombat batch correction
                    corrected_group_df = pycombat(group_df, batch_list, mod=healthy_list, par_prior=True)

                    # Assign batch corrected expression matrix to group adata
                    group_adata.X = corrected_group_df.T
                    # Add group adata to dictionary
                    batch_corrected_adata[group] = group_adata

                # Get the error if it fails
                except Exception as e:
                    print(e)   
                    raise ValueError('Error in batch correction. Check the error above.')

            # Concatenate all batch corrected groups
            batch_corrected_adata = ad.concat(batch_corrected_adata.values())

            # Get data matrix and category_df from batch corrected adata
            corrected_data_matrix = pd.DataFrame(batch_corrected_adata.X.T, index=data_matrix.index, columns=batch_corrected_adata.obs.index)
            corrected_category_df = batch_corrected_adata.obs

            return corrected_data_matrix, corrected_category_df

        # TODO: Handle the cases dataset= tcga, gtex or both

        # Start taking time
        start_time = time.time()

        # If healthy column is not present in category_df, add it
        if 'healthy' not in category_df.columns:
            category_df['healthy'] = category_df['lab_txt'].str.contains('GTEX')

        # Manage the levels of processing
        if wang_level == 0:
            # Do nothing
            print('No Wang processing done...')
            return data_matrix, category_df
        
        if wang_level >= 1:
            print('Performing wang level 1 processing: Removal of not paired samples...')
            data_matrix, category_df = wang_sample_filtering(data_matrix, category_df)
        
        if wang_level >= 2:
            print('Performing wang level 2 processing: Quantile normalization and log2(x+1) transform...')
            data_matrix = wang_quantile_normalization(data_matrix)
        
        if wang_level >= 3:
            print('Performing wang level 3 processing: ComBat batch correction...')
            data_matrix, category_df = wang_combat(data_matrix, category_df)

        # Print the time it took to wang process the data
        print(f'Wang processing took {time.time() - start_time:.2f} seconds.')
        
        return data_matrix, category_df

    # This function performs a data normalization by batches (GTEX or TCGA) 
    def batch_normalize(self):
        if self.batch_normalization==False:
            print('Did not perform batch normalization...')
            return
        else:
            print('Batch normalizing matrix data...')
            start = time.time()
            # Get the identifiers of the samples in each subset
            gtex_samples = self.label_df[self.label_df['is_tcga']==False].index
            tcga_samples = self.label_df[self.label_df['is_tcga']==True].index
            
            # Get the data matrices of each subset
            gtex_data = self.gene_filtered_data_matrix[gtex_samples]
            tcga_data = self.gene_filtered_data_matrix[tcga_samples]

            # Get index and columns of the data matrices
            gtex_index = gtex_data.index
            tcga_index = tcga_data.index
            gtex_columns = gtex_data.columns
            tcga_columns = tcga_data.columns

            # Apply standardization to each subset
            normalized_gtex_data = StandardScaler().fit_transform(gtex_data.T).T
            normalized_tcga_data = StandardScaler().fit_transform(tcga_data.T).T
            
            # Create dataframes from the normalized data
            normalized_gtex_df = pd.DataFrame(normalized_gtex_data, index=gtex_index, columns=gtex_columns)
            normalized_tcga_df = pd.DataFrame(normalized_tcga_data, index=tcga_index, columns=tcga_columns)

            # Concatenate the normalized dataframes
            normalized_joint = pd.concat([normalized_gtex_df, normalized_tcga_df], axis=1)

            # Sort columns of normalized joint
            normalized_joint = normalized_joint.T.sort_index().T

            # Replace NaNs generated by std division to 0's
            self.gene_filtered_data_matrix = normalized_joint.fillna(0.0)
            end = time.time()
            print(f'It took {round(end-start, 2)} s to batch normalize the data.')    

    # This function uses self.binary_dict to modify self.label_df and self.lab_txt_2_lab_num to make the labels binary
    def make_binary_problem(self):
        # If binary_dict is not specified, do not make binary
        if self.binary_dict == {}:
            print("No binary problem specified.")
            return
        # If binary_dict is specified, make binary
        else:
            self.lab_txt_2_lab_num = self.binary_dict
            # Define numeric labels from the textual labels in self.label_df
            self.label_df["lab_num"] = self.label_df["lab_txt"].map(self.lab_txt_2_lab_num)   
            print("Made binary problem.")
            print(f"Number of samples in class 0: {len(self.label_df[self.label_df['lab_num'] == 0])}, number of samples in class 1: {len(self.label_df[self.label_df['lab_num'] == 1])}")
            return

    # Get the indexes of of each fold
    def get_k_fold_indexes(self):
        # Get dummy x data and real y values
        dummy_x = np.zeros(len(self.label_df["lab_num"]))
        global_y = np.ravel(self.label_df["lab_num"].values)
        counter = 0
        k_fold_indexes = {}
        # Add k-fold indexes to dictionary
        for train_index, test_index in self.k_fold_cross_validator.split(dummy_x, global_y):
            k_fold_indexes[counter] = {'train_index': train_index, 'test_index': test_index}
            counter = counter + 1
        return k_fold_indexes

    # This function gets the dataloaders for the train, val and test sets
    def get_dataloaders(self, batch_size=100, fold=0):

        # Get train and test indexes depending on the fold number
        train_index, test_index = self.k_fold_indexes[fold]['train_index'], self.k_fold_indexes[fold]['test_index']
        
        # Get train and test matrices and groundtruth
        train_matrix, test_matrix = self.gene_filtered_data_matrix.iloc[:, train_index], self.gene_filtered_data_matrix.iloc[:, test_index]
        train_gt, test_gt = self.label_df.iloc[train_index]['lab_num'], self.label_df.iloc[test_index]['lab_num'] 
        
        # Pass matrices to tensors and transpose them to have samples in rows and genes in columns
        x_train = torch.Tensor(train_matrix.T.values).type(torch.float)
        x_test = torch.Tensor(test_matrix.T.values).type(torch.float)
        
        # Cast labels as tensors
        y_train = torch.Tensor(train_gt.values).type(torch.long)
        y_test = torch.Tensor(test_gt.values).type(torch.long)
        
        # Define train, val and test datasets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    # This function uses self.label_df to split the data into train and test sets depending on the fold number
    def get_numpy_split(self, fold=0):
        # Get train and test indexes depending on the fold number
        train_index, test_index = self.k_fold_indexes[fold]['train_index'], self.k_fold_indexes[fold]['test_index']
        # Get train and test matrices and groundtruth
        train_matrix, test_matrix = self.gene_filtered_data_matrix.iloc[:, train_index], self.gene_filtered_data_matrix.iloc[:, test_index]
        train_gt, test_gt = self.label_df.iloc[train_index]['lab_num'], self.label_df.iloc[test_index]['lab_num'] 
        # Declare x and y dictionary
        split_dict = {'x': {'train': train_matrix, 'test': test_matrix}, 'y': {'train': train_gt, 'test': test_gt}}
        # Return split dictionary
        return split_dict

    # This is just as get_numpy_split but it returns the annotations of the batch (0: gtex, 1: tcga)
    def get_batch_split(self, fold=0):
        # Get train and test indexes depending on the fold number
        train_index, test_index = self.k_fold_indexes[fold]['train_index'], self.k_fold_indexes[fold]['test_index']
        # Get train and test matrices and groundtruth
        train_matrix, test_matrix = self.gene_filtered_data_matrix.iloc[:, train_index], self.gene_filtered_data_matrix.iloc[:, test_index]
        train_gt, test_gt = self.label_df.iloc[train_index]['is_tcga'].astype(int), self.label_df.iloc[test_index]['is_tcga'].astype(int) 
        # Declare x and y dictionary
        split_dict = {'x': {'train': train_matrix, 'test': test_matrix}, 'y': {'train': train_gt, 'test': test_gt}}
        # Return split dictionary
        return split_dict        

    # This function augments the self.label_df dataframe with textual and numeric hong annotations.
    # The function returns a dictionary that goes from standard numeric annotations to hong tuple annotations
    # and other dictionary that goes from hong tuple annotations (numeric) to standard int annotations. 
    def compute_hong_annotations(self):
        # Read lab_txt_2_tissue mapper
        with open(os.path.join(self.path, "mappers", "id_2_tissue_mapper.json"), "r") as f:
            lab_txt_2_tissue = json.load(f)
        
        # Define some extra lab_2_tissue entries
        extra_entries = {   'GTEX-FAL_TUB': 'Fallopian tube',
                            'GTEX-HEA':     'Heart',
                            'GTEX-NER':     'Nerve',
                            'GTEX-SAL_GLA': 'Salivary gland',
                            'GTEX-SMA_INT': 'Small intestine',
                            'GTEX-SPL':     'Spleen',
                            'GTEX-VAG':     'Vagina',
                            'TCGA-DLBC':    'Lymphatic',
                            'TCGA-HNSC':    'Mucosal Epithelium',
                            'TCGA-THYM':    'Thymus',
                            'TCGA-UVM':     'Uvea'}

        # Update lab_txt_2_tissue with extra entries
        lab_txt_2_tissue.update(extra_entries)

        # get reversed tissue_2_lab_txt_tcga dictionary just for tcga labels.
        # If there are multiple lab_txt for a tissue then they are stored in a list 
        tissue_2_lab_txt_tcga = {}
        for key, value in lab_txt_2_tissue.items():
            if 'TCGA' in key:
                tissue_2_lab_txt_tcga.setdefault(value, []).append(key)

        # Make dictionary that maps every lab_txt to a subtype. If there is only one cancer
        # type for a given tissue then 'No Subtype' is assigned
        lab_txt_2_subtype = {}
        for lab, tissue in lab_txt_2_tissue.items():
            if ('TCGA' in lab) and (len(tissue_2_lab_txt_tcga[tissue])>1):
                lab_txt_2_subtype[lab] = lab
            else:
                lab_txt_2_subtype[lab] = 'No Subtype'

        # Add textual (cancer, tissue and subtype) labels to self.label_df
        self.label_df['hong_cancer'] = self.label_df['lab_txt'].str.contains('TCGA')
        self.label_df['hong_tissue'] = self.label_df['lab_txt'].map(lab_txt_2_tissue)
        self.label_df['hong_subtype'] = self.label_df['lab_txt'].map(lab_txt_2_subtype)

        # Get sorted lists of the present textual tissues and subtypes
        sorted_tissues = sorted(self.label_df['hong_tissue'].unique())
        sorted_subtypes = sorted(self.label_df['hong_subtype'].unique())
        sorted_subtypes.remove('No Subtype') # Remove No subtype to add it later with a -1 numeric label 
        
        # Compute numeric dictionaries of each textual hong annotation
        cancer_2_num = {True:1, False:0}
        tissue_2_num = {key:num for num, key in enumerate(sorted_tissues)}
        subtype_2_num = {key:num for num, key in enumerate(sorted_subtypes)}
        subtype_2_num['No Subtype'] = -1 # Important: -1 means that there is no subtype for that sample

        # Get a hong annotation list of tuples
        hong_annot = list(zip(  self.label_df['hong_cancer'].map(cancer_2_num),
                                self.label_df['hong_tissue'].map(tissue_2_num),
                                self.label_df['hong_subtype'].map(subtype_2_num)))
        
        self.label_df['hong_annot'] = hong_annot # Assign hong annotation column

        # Get dictionaries from standard to hong annotations and viceversa
        standard_2_hong_annot = dict(zip(self.label_df['lab_num'], self.label_df['hong_annot']))
        hong_2_standard_annot = {val: key for key, val in standard_2_hong_annot.items()}

        return standard_2_hong_annot, hong_2_standard_annot

    # pca=-1 to not use pca.
    # TODO: Write a good documentation
    def get_hong_dataloaders(self, batch_size_multitask, batch_size_subtype, pca=2000, fold=0):

        # Get train and test indexes depending on the fold number
        train_index, test_index = self.k_fold_indexes[fold]['train_index'], self.k_fold_indexes[fold]['test_index']

        # Declare the processed data
        processed_data = self.gene_filtered_data_matrix.T.values

        # If a pca int bigger than 0 i specified then transform the original data
        if pca>0:
            print('Reducing dimensions with PCA may take approx 3 minutes...')
            pca_reductor = PCA(n_components=pca, random_state=0)
            processed_data = pca_reductor.fit_transform(processed_data)
        else:
            pass

        # Compute hong annotations and get the train and test ground truths
        standard_2_hong_annot, hong_2_standard_annot = self.compute_hong_annotations()
        global_gt = np.array([*self.label_df['hong_annot'].values])
        train_gt, test_gt = global_gt[train_index, :], global_gt[test_index, :]

        # Get train and test matrix
        train_matrix, test_matrix = processed_data[train_index, :], processed_data[test_index, :]
        
        # Cast labels as tensors
        y_train = torch.Tensor(train_gt).type(torch.long)
        y_test = torch.Tensor(test_gt).type(torch.long)

        # Pass matrices to tensors. samples in rows and genes in columns
        x_train = torch.Tensor(train_matrix).type(torch.float)
        x_test = torch.Tensor(test_matrix).type(torch.float)

        # Define train, val and test datasets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

        # Find indexes of data that have a subtype
        valid_subtype_train_indexes = y_train[:,2] != -1 
        valid_subtype_test_indexes = y_test[:,2] != -1

        # Get x_train/test for samples with subtype
        valid_subtype_x_train = x_train[valid_subtype_train_indexes, :]
        valid_subtype_x_test = x_test[valid_subtype_test_indexes, :]

        # Get y_train/test for samples with subtype
        valid_subtype_y_train = y_train[valid_subtype_train_indexes, :]
        valid_subtype_y_test = y_test[valid_subtype_test_indexes, :]

        # Declare subtype datasets with only samples that have subtypes
        train_subtype_dataset = TensorDataset(valid_subtype_x_train, valid_subtype_y_train) 
        test_subtype_dataset = TensorDataset(valid_subtype_x_test, valid_subtype_y_test) 

        # Create dataloaders. Separated for multitask and subtype  
        train_loader_multitask = DataLoader(train_dataset, batch_size=batch_size_multitask, shuffle=True)
        test_loader_multitask = DataLoader(test_dataset, batch_size=batch_size_multitask, shuffle=False) # Shuffle in test is unnecessary
        
        train_loader_subtype = DataLoader(train_subtype_dataset, batch_size=batch_size_subtype, shuffle=True)
        test_loader_subtype = DataLoader(test_subtype_dataset, batch_size=batch_size_subtype, shuffle=False) # Shuffle in test is unnecessary

        # Create return dict for all dataloaders and annotation dictionaries
        return_dict = { 'multitask':        (train_loader_multitask, test_loader_multitask),
                        'subtype':          (train_loader_subtype, test_loader_subtype),
                        'annot':            (standard_2_hong_annot, hong_2_standard_annot),
                        'valid_index':      (valid_subtype_train_indexes, valid_subtype_test_indexes),
                        'test_standard_gt': self.label_df['lab_num'].values[test_index]}    
        return return_dict

    # This function plots a 2X2 figure with the histograms of mean expression and standard deviation before and after filtering
    def plot_filtering_histograms(self, filtering_info_df):        
        # Make a figure
        fig, axes = plt.subplots(2, 2, figsize = (18, 12))
        fig.suptitle("Filtering of Mean > " +str(self.mean_thr) + " and Standard Deviation > " + str(self.std_thr) , fontsize = 30)
        # Variable to adjust display height of the histograms
        max_hist = np.zeros((2, 2))

        # Plot the histograms of mean and standard deviation before filtering
        n, _, _ = axes[0, 0].hist(filtering_info_df["joint_mean"], bins = 50, color = "k", density=True)
        max_hist[0, 0] = np.max(n)
        axes[0, 0].set_title("Before filtering", fontsize = 26)        
        n, _, _ = axes[1, 0].hist(filtering_info_df["joint_std"], bins = 50, color = "k", density=True)
        max_hist[1, 0] = np.max(n)
        # Plot the histograms of mean and standard deviation after filtering
        n, _, _ = axes[0, 1].hist(filtering_info_df["joint_mean"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
        max_hist[0, 1] = np.max(n)
        axes[0, 1].set_title("After filtering", fontsize = 26)
        n, _, _ = axes[1, 1].hist(filtering_info_df["joint_std"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
        max_hist[1, 1] = np.max(n)

        # Format axes
        for i in range(2):
            for j in range(2):
                axes[i, j].set_ylabel("Density", fontsize = 16)
                axes[i, j].tick_params(labelsize = 14)
                axes[i, j].grid(True)
                axes[i, j].set_axisbelow(True)
                axes[i, j].set_ylim(0, max_hist[i, j] * 1.1)
                # Handle mean expression plots
                if i == 0:
                    axes[i, j].set_xlabel("Mean expression", fontsize = 16)
                    axes[i, j].set_xlim(filtering_info_df["joint_mean"].min(), filtering_info_df["joint_mean"].max())
                    axes[i, j].plot([self.mean_thr, self.mean_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")
                # Handle standard deviation plots
                else:
                    axes[i, j].set_xlabel("Standard deviation", fontsize = 16)
                    axes[i, j].set_xlim(filtering_info_df["joint_std"].min(), filtering_info_df["joint_std"].max())
                    axes[i, j].plot([self.std_thr, self.std_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")

        # Save the figure
        fig.savefig(os.path.join(self.dataset_info_path, "filtering_histograms.png"), dpi = 300)
        plt.close(fig)

    def plot_dim_reduction(self):
        
        valid_samples = self.gene_filtered_data_matrix.columns
        valid_genes = self.filtered_gene_list
        bool_sample_index = self.matrix_data.columns.isin(valid_samples)
        bool_gene_index = self.matrix_data.index.isin(valid_genes)
        print('Filtering raw data to valid genes and samples...')
        raw_data = self.matrix_data.loc[bool_gene_index, bool_sample_index].T.sort_index()
        processed_data = self.gene_filtered_data_matrix.T.sort_index()
        # Get and sort metadata
        meta_df = self.label_df
        meta_df = meta_df.sort_index()

        if (not os.path.exists(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'))) or self.force_compute:
            
            print(f'Computing dimensionality reduction and saving it to {os.path.join(self.dataset_info_path, "dimensionality_reduction.pkl")}...')
            
            print('Reducing dimensions with PCA...')
            raw_pca = PCA(n_components=2, random_state=0)
            processed_pca = PCA(n_components=2, random_state=0)
            r_raw_pca = raw_pca.fit_transform(raw_data)
            r_processed_pca = processed_pca.fit_transform(processed_data)

            # If this shows a warning with OpenBlast you can solve it using this GitHub issue https://github.com/ultralytics/yolov5/issues/2863
            # You just need to write in terminal "export OMP_NUM_THREADS=1"
            print('Reducing dimensions with TSNE...')
            raw_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='random', n_jobs=-1, verbose=2)       # learning_rate and init were set due to a future warning
            processed_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='random', n_jobs=-1, verbose=2)
            r_raw_tsne = raw_tsne.fit_transform(raw_data)
            r_processed_tsne = processed_tsne.fit_transform(processed_data)

            reduced_dict = {'raw_pca': r_raw_pca,               'raw_tsne': r_raw_tsne,           
                            'processed_pca': r_processed_pca,   'processed_tsne': r_processed_tsne}

            with open(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'), 'wb') as f:
                pkl.dump(reduced_dict, f)
        
        else:
            
            print(f'Loading dimensionality reduction from {os.path.join(self.dataset_info_path, "dimensionality_reduction.pkl")}...')
            
            with open(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'), 'rb') as f:
                reduced_dict = pkl.load(f)
        

        def plot_dim_reduction(reduced_dict, meta_df, color_type='tcga', cmap=None):


            # Load id_2_tissue mapper from file
            with open(os.path.join(self.path, "mappers", "id_2_tissue_mapper.json"), "r") as f:
                id_2_tissue_mapper = json.load(f)

            meta_df['tissue'] = meta_df['lab_txt'].map(id_2_tissue_mapper)

            # Get dictionaries to have different options to colorize the scatter points
            tcga_dict = {True: 1, False: 0}
            tissue_dict = {tissue: i/len(meta_df.tissue.unique()) for i, tissue in enumerate(sorted(meta_df.tissue.unique()))}
            class_dict = {cl: i/len(meta_df.lab_txt.unique()) for i, cl in enumerate(sorted(meta_df.lab_txt.unique()))}

            # Define color map
            if cmap is None:
                d_colors = ["black", "darkcyan"]
                cmap = LinearSegmentedColormap.from_list("candle_cmap", d_colors)
            else:
                cmap = get_cmap(cmap)
            
            # Compute color values
            if color_type == 'tcga':
                meta_df['color'] = meta_df['is_tcga'].map(tcga_dict)
            elif color_type == 'tissue':
                meta_df['color'] = meta_df['tissue'].map(tissue_dict)
            elif color_type == 'class':
                meta_df['color'] = meta_df['lab_txt'].map(class_dict)

            # Plot figure
            fig, ax = plt.subplots(2, 2)

            ax[0, 0].scatter(reduced_dict['raw_pca'][:,0],          reduced_dict['raw_pca'][:,1],           c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
            ax[1, 0].scatter(reduced_dict['processed_pca'][:,0],    reduced_dict['processed_pca'][:,1],     c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
            ax[0, 1].scatter(reduced_dict['raw_tsne'][:,0],         reduced_dict['raw_tsne'][:,1],          c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
            ax[1, 1].scatter(reduced_dict['processed_tsne'][:,0],   reduced_dict['processed_tsne'][:,1],    c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)


            label_dict = {0: 'PC', 1: 'T-SNE'}
            title_dict = {0: 'PCA', 1: 'T-SNE'}

            for i in range(ax.shape[0]):
                for j in range(ax.shape[1]):
                    ax[i,j].spines.right.set_visible(False) 
                    ax[i,j].spines.top.set_visible(False)
                    ax[i,j].set_xlabel(f'{label_dict[j]}1')
                    ax[i,j].set_ylabel(f'{label_dict[j]}2')
                    ax[i,j].set_title(f'{title_dict[j]} of Raw Data' if i==0 else f'{title_dict[j]} of Processed Data')

            # Set the legend
            if (color_type == 'tcga'):
                handles = [mpatches.Patch(facecolor=cmap(0.0), edgecolor='black', label='GTEx'),
                           mpatches.Patch(facecolor=cmap(1.0), edgecolor='black', label='TCGA')]
                fig.set_size_inches(13, 7)
                ax[0, 2].legend(handles=handles, loc='center left',bbox_to_anchor=(1.15, 0.49))
                plt.tight_layout()

            elif (color_type == 'tissue'):
                handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'{key} Tissue') for key, val in tissue_dict.items()]
                fig.set_size_inches(15, 7)
                fig.legend(handles=handles, loc=7)
                fig.tight_layout()
                fig.subplots_adjust(right=0.83)
                

            elif (color_type == 'class'):
                handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'{key}') for key, val in class_dict.items()]
                fig.set_size_inches(18, 8)
                fig.legend(handles=handles, loc=7, ncol=2)
                fig.tight_layout()
                fig.subplots_adjust(right=0.75)

            
            else:
                raise ValueError('Invalid color type...')

            # Save
            fig.savefig(os.path.join(self.dataset_info_path, f'dim_reduction_{color_type}.png'), dpi=300)
            plt.close()

        # Plot dimensionality reduction with the three different color styles
        plot_dim_reduction(reduced_dict, meta_df, color_type='tcga')
        plot_dim_reduction(reduced_dict, meta_df, color_type='tissue', cmap='brg')
        plot_dim_reduction(reduced_dict, meta_df, color_type='class', cmap='brg')

# TODO: Make adequate documentation
def map_ensg_to_symbol(matrix_data):

    print('Mapping ensemble gene IDs to gene symbol...')
    start = time.time()

    ### Undo the log2 transformation
    print('Reversing log2(x+offset) transformation...')
    glob_min = matrix_data.min().min()
    offset = np.power(2, glob_min)
    matrix_data = np.power(2, matrix_data) - offset

    print(f'Computed offset: {offset}')

    ### Map ensemble gene id to gene symbol
    gene_names = pd.read_csv(os.path.join('data', "gene_names.csv"))                # Load the gene names csv file
    matrix_data.index = matrix_data.index.str.split(".").str[0]                     # For all indexes in matrix_data remove substring after the first dot (This just indicates the version)
    print('Filtering out genes not in gene name notation mapping...')
    matrix_data = matrix_data.loc[matrix_data.index.isin(gene_names['ensembl_id'])] # Filter out genes not in gene name notation mapping
    matrix_data['ensembl_id'] = matrix_data.index                                   # Add a column with the ensemble gene id to the matrix_data
    gene_names_dict = dict(zip(gene_names['ensembl_id'], gene_names['symbol']))     # Get a dict with the ensemble gene id as key and the gene symbol as value
    matrix_data['gene_symbol'] = matrix_data['ensembl_id'].map(gene_names_dict)     # Map the ensemble gene id to the gene symbol
    matrix_data.drop('ensembl_id', axis=1, inplace=True)                            # Delete the ensemble gene id column

    ### Deal with duplicated gene symbols
    duplicated_genes = matrix_data[matrix_data['gene_symbol'].duplicated(keep=False)] # Get subset of genes with duplicated gene symbols
    duplicated_genes = duplicated_genes.groupby('gene_symbol').sum()                  # Group duplicated genes by gene symbol by summing their expression values
    matrix_data = matrix_data[~matrix_data['gene_symbol'].duplicated(keep=False)]     # Remove duplicated genes from matrix_data
    matrix_data.set_index('gene_symbol', inplace=True)                                # Reindex matrix_data by gene_symbol
    matrix_data = pd.concat([matrix_data, duplicated_genes])                          # Add duplicated genes to matrix_data

    # Print number of duplicated genes that had to be corrected
    print(f'{len(duplicated_genes)} gene symbols had multiple ENSG ids and were corrected.')

    # Assert that there are no duplicated genes and print message
    assert matrix_data.index.duplicated().sum() == 0 , 'There are duplicated genes in the matrix_data'

    # Sort matrix_data by gene_symbol
    matrix_data.sort_index(inplace=True)

    ### Redo the log2 transformation
    print('Redoing log2(x+offset) transformation...')
    matrix_data = np.log2(matrix_data + offset)

    print(f'Finished mapping ensemble gene id to gene symbol in {time.time()-start:.2f} seconds.')

    return matrix_data


# Reading functions for all datasets
def read_toil(path, force_compute):
        """
        Reads data from the Toil data set with root path and returns matrix_data, categories and phenotypes dataframes.

        Args:
            path (str): Root path of the Toil data set.

        Returns:
            matrix_data (pd.dataframe): Matrix data of the Toil data set. Columns are samples and rows are genes.
            categories (pd.dataframe): Categories of the Toil data set. Rows are samples.
        """
        start = time.time()
        matrix_data = pd.read_feather(os.path.join(path, "data_matrix.feather"))
        categories = pd.read_csv(os.path.join(path, "phenotypes.csv"), encoding = "cp1252")
        # Delete the first column of categories and phenotypes
        categories = categories.drop(categories.columns[0], axis = 1)
        # Set the first column of matrix_data, categories and phenotypes as index
        matrix_data.set_index(matrix_data.columns[0], inplace = True)
        categories.set_index(categories.columns[0], inplace = True)
        # Delete the rows with nan values
        categories.dropna(inplace=True)
        # Filter out all TARGET samples
        matrix_data = matrix_data.iloc[:, ~matrix_data.columns.str.contains("TARGET")]
        categories = categories[~(categories['_study']=='TARGET')]
        # Add is_tcga column to categories and phenotypes
        categories['is_tcga'] = categories['_study'] == 'TCGA'
        # Ensure that all data in matrix_data are in categories
        matrix_data = matrix_data.loc[:, matrix_data.columns.isin(categories.index)]
        # Load the phenotypes to textual labels dictionary.
        with open(os.path.join(path, "mappers", "phenotype_mapper.json"), "r") as f:
            pheno_2_lab_txt = json.load(f)
        # Get initial textual labels
        categories['lab_txt'] = categories['detailed_category'].map(pheno_2_lab_txt)
        # Refine labels in healthy TCGA samples
        # Get normal mapper
        with open(os.path.join(path, "mappers", "normal_tcga_2_gtex_mapper.json"), "r") as f:
            normal_tcga_2_gtex = json.load(f)
        # Get TCGA healthy samples bool index
        tcga_healthy_bool = (categories['_sample_type']=="Solid Tissue Normal") & (categories['is_tcga'])
        # Map correctly healthy TCGA
        categories.loc[tcga_healthy_bool, 'lab_txt'] = categories.loc[tcga_healthy_bool, "_primary_site"].map(normal_tcga_2_gtex)
        # Sort matrix data columns by samples name
        matrix_data = matrix_data.reindex(sorted(matrix_data.columns), axis=1)
        # Sort categories dataframe by sample name
        categories.sort_index(inplace=True)
        end = time.time()
        print("Time to load data: {} s".format(round(end - start, 3)))

        # Map ensemble gene id to gene symbol
        matrix_data = map_ensg_to_symbol(matrix_data)

        return matrix_data, categories

def read_wang(path, force_compute):
    """
    Read the Wang data set with root path and returns matrix_data and categories dataframes. The 
    matrix_data dataframe is already TPM normalized and log2(x+1) transformed.
    """
    # Define helper functions
    # This function unzips the raw downloaded data from 
    def unzip_data(path, force_compute):
        final_data_path = os.path.join(path, 'original_data')
        # Do nothing if unzipped folder already exists
        if os.path.exists(final_data_path) and (force_compute==False):
            print('Files already unzipped...')
            return
        # Unzip data if original_data does not exist
        else:
            print('Unzipping files this may take some minutes...')
            zipped_path = os.path.join(path, 'raw_data.zip')
            unzipped_folder = os.path.join(path, 'raw_data_unzipped')
            final_data_path = os.path.join(path, 'original_data')

            with zipfile.ZipFile(zipped_path, 'r') as zip_ref:
                zip_ref.extractall(unzipped_folder)
            
            classes_paths = os.listdir(unzipped_folder)
            
            # Make final directory
            os.makedirs(final_data_path, exist_ok=True)
            # Cycle to unzip original data
            for i in tqdm(range(len(classes_paths))):
                class_path = classes_paths[i]
                final_file_name = class_path[:-3]
                
                with gzip.open(os.path.join(unzipped_folder, class_path), 'rb') as f_in:
                    with open(os.path.join(final_data_path, final_file_name), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            # Remove temporal folder
            shutil.rmtree(unzipped_folder)

    # This helper function receives the file name of a class and returns a valid textual label
    def get_label_from_name(name):
        str_list = name[:-4].split('-')
        if len(str_list) == 5:
            label = str_list[-2]+'-'+str_list[-1]+'-'+str_list[0]
        elif len(str_list) == 4:
            label = str_list[-1]+'-'+str_list[0]
        else:
            raise ValueError('The name of the original file is not adequate.')
        label = label.upper()

        return label


    def tpm_transform(data: pd.DataFrame) -> pd.DataFrame:
        """
        This function transforms the counts in a dataframe to TPM. The function returns the transformed dataframe object.
        It also removes genes that are not fount in the gtf annotation file.

        Args:
            data (pd.DataFrame): The dataframe object to transform.

        Returns:
            pd.Dataframe: The transformed dataframe object with TPM values.
        """
        # Get the number of genes before filtering
        initial_genes = data.shape[0]

        # FIXME: this should work with gencode.v19.annotation but for now it is working with GCF_000001405.40
        # Unzip the data in annotations folder if it is not already unzipped
        # if not os.path.exists(os.path.join('data', 'annotations', 'gencode.v19.annotation.gtf')):
        #    with gzip.open(os.path.join('data', 'annotations', 'gencode.v19.annotation.gtf.gz'), 'rb') as f_in:
        #         with open(os.path.join('data', 'annotations', 'gencode.v19.annotation.gtf'), 'wb') as f_out:
        #             shutil.copyfileobj(f_in, f_out)
        
        # Define gtf path
        # gtf_path = os.path.join('data', 'annotations', 'gencode.v19.annotation.gtf')

        # Unzip the data in annotations folder if it is not already unzipped
        if not os.path.exists(os.path.join('data', 'annotations', 'GCF_000001405.40')):
            with zipfile.ZipFile(os.path.join('data', 'annotations', 'GCF_000001405.40.zip'), 'r') as zip_ref:
                zip_ref.extractall(os.path.join('data', 'annotations', 'GCF_000001405.40'))
        
        # Define gtf path
        gtf_path = os.path.join('data', 'annotations', 'GCF_000001405.40', 'ncbi_dataset', 'data', 'GCF_000001405.40', 'genomic.gtf')

        # Get the TPM transform object
        tpm_transformation = TPM(gtf=gtf_path).set_output(transform="pandas")

        # Get the counts matrix and transpose it
        counts = data.copy().T

        # Transform the counts to TPM
        start = time.time()
        tpm = tpm_transformation.fit_transform(counts)
        end = time.time()
        print(f'TPM transformation took {end - start:.2f} seconds')

        # Get the genes that do not have nan values in the TPM matrix
        mask = ~tpm.isna().any(axis=0)

        # Filter the genes
        tpm = tpm.loc[:, mask]

        # Transpose the TPM matrix again
        tpm = tpm.T

        # Print filtering results
        rem_genes = initial_genes - tpm.shape[0]
        rem_genes_pct = (rem_genes) / initial_genes * 100
        print(f'Number of genes removed by TPM transformation: {rem_genes}/{initial_genes} ({rem_genes_pct:.2f}%)')

        # Return the transformed AnnData object
        return tpm


    # Unzip data in case it is needed
    unzip_data(path, force_compute)

    # If processed data directory does not exist read, merge and save complete data
    if not os.path.exists(os.path.join(path, 'processed_data')) or (force_compute == True):
        
        print(f'Reading data from {os.path.join(path, "original_data")}')
        start = time.time()
        data_path = os.path.join(path, 'original_data')
        # Declare list of paths where each class is hosted
        classes_paths = os.listdir(data_path)

        for i in tqdm(range(len(classes_paths))):
            class_file = classes_paths[i] # Get file name
            act_df = pd.read_table(os.path.join(data_path, class_file), delimiter='\t') # Read file
            
            # Perform minor modification in act_df
            act_df = act_df.set_index('Hugo_Symbol') 
            del act_df['Entrez_Gene_Id']
            
            # Filter out genes in act_df that are not in all classes 
            valid_gene_index = act_df.index if i==0 else valid_gene_index.intersection(act_df.index)
            act_df = act_df.loc[valid_gene_index, :]

            act_category = get_label_from_name(class_file) # Get label names from file names
            act_category_df = pd.DataFrame({'sample':act_df.columns, 'original_lab': act_category}) # Put original label name
            
            # Join iteratively data matrices
            data_matrix = act_df if i==0 else data_matrix.join(act_df)
            data_matrix = data_matrix.loc[valid_gene_index, :] # Ensure data matrix just has common genes in all classes
            category_df = act_category_df if i==0 else pd.concat([category_df, act_category_df], axis=0) # Join category dataframes

        # Sort Genes and samples in data matrix
        data_matrix.sort_index(inplace=True) # Sort genes
        data_matrix = data_matrix.T.sort_index().T # Sort samples
        # Set samples as index and sort category dataframe
        category_df.set_index('sample', inplace=True)
        category_df.sort_index(inplace=True)


        # Add a binary column to category_df indicating if the samples is from the TCGA
        category_df['is_tcga'] = category_df['original_lab'].str.contains('TCGA')

        # Loads standard label mapper
        with open(os.path.join(path, "mappers", "wang_standard_label_mapper.json"), "r") as f:
            standard_label_mapper = json.load(f)

        # Add a column with the standard labels shared between all datasets (Toil, Wang and Recount3)
        category_df['lab_txt'] = category_df['original_lab'].map(standard_label_mapper)

        # Print time that was needed to read the data
        end = time.time()
        print(f'Time to read data: {round(end-start,2)} s')

        # Perform TPM transformation
        data_matrix = tpm_transform(data_matrix)

        # Log2(x+1) transform the data
        tqdm.pandas(desc="Computing Log2(x+1) transform")
        data_matrix = data_matrix.progress_apply(lambda x: np.log2(x+1), axis=1)
        
        # Sort matrix data columns by samples name
        data_matrix = data_matrix.reindex(sorted(data_matrix.columns), axis=1)
        # Sort categories dataframe by sample name
        category_df.sort_index(inplace=True)

        # Reset index to save in feather file
        data_matrix.reset_index(inplace=True)

        # Change the name of the index column to Hugo_Symbol
        data_matrix.rename(columns={'index': 'Hugo_Symbol'}, inplace=True)

        print(f'Saving processed data to {os.path.join(path, "processed_data")}')
        os.makedirs(os.path.join(path, 'processed_data'), exist_ok=True)
        data_matrix.to_feather(os.path.join(path, 'processed_data', 'data_matrix.feather'))
        category_df.to_csv(os.path.join(path, 'processed_data', 'data_category.csv'))

        # Set index again for next steps
        data_matrix.set_index('Hugo_Symbol', inplace=True)

        
    # If the data is already merged and stored load it from file
    else:
        print(f'Loading processed data from {os.path.join(path, "processed_data")}')
        start = time.time()
        data_matrix = pd.read_feather(os.path.join(path, 'processed_data', 'data_matrix.feather'))
        category_df = pd.read_csv(os.path.join(path, 'processed_data', 'data_category.csv'), index_col='sample')
        end = time.time()
        print(f'Time to read data: {round(end-start,2)} s')
        data_matrix.set_index('Hugo_Symbol', inplace=True)

    return data_matrix, category_df

# FIXME: Make sure that this data is in TPM
def read_recount3(path, force_compute):
        """
        Reads data from the Recount3 data set with root path and returns matrix_data and categories dataframes.

        Returns:
            matrix_data (pd.dataframe): Matrix data of the Recount3 data set. Columns are samples and rows are genes.
            categories (pd.dataframe): Categories of the Recount3 data set. Rows are samples. Columns are 'lab_txt', 'is_tcga' and 'healthy'
            gene_meta (pd.DataFrame): Useful information about the available genes.
        """
        start = time.time()

        if (not os.path.exists(os.path.join(path, "processed_data"))) or force_compute:
            print(f'Reading data from {os.path.join(path, "original_data")} and performing processing and transformations...')
            # Read all data
            matrix_data = pd.read_feather(os.path.join(path, 'original_data', "data_matrix.feather"))
            gtex_meta = pd.read_csv(os.path.join(path, 'original_data', "gtex_metadata.csv"), low_memory=False)
            tcga_meta = pd.read_csv(os.path.join(path, 'original_data', "tcga_metadata.csv"), low_memory=False)
            gene_meta = pd.read_csv(os.path.join(path, 'original_data', "gene_metadata.csv"), low_memory=False)
            
            # Process TCGA metadata #############
            # Filter just important TCGA metadata
            # TODO: Sample identifiers here are weird for TCGA metadata and data matrix. It works but it would be better to standardize identifiers with wang and toil
            tcga_meta = tcga_meta[['Unnamed: 0', 'tcga.gdc_cases.project.project_id', 'tcga.gdc_cases.project.primary_site', 'tcga.cgc_sample_sample_type']]
            # Rename columns
            tcga_meta.columns = ['sample', 'lab_tcga', 'tissue', 'sample_type']
            # Get a column of the tcga_meta indicating if the data is healthy
            tcga_meta['healthy'] = tcga_meta['sample_type'] == 'Solid Tissue Normal'
            # Reset index
            tcga_meta.set_index('sample', inplace=True)
            # Filter out rows with NaNs
            tcga_meta.dropna(inplace=True)
            # Add is_tcga column
            tcga_meta['is_tcga'] = True

            # Get standard lab_txt labels from tcga_meta #######
            # Initialize the lab_txt column as the TCGA projects
            tcga_meta['lab_txt'] = tcga_meta['lab_tcga']
            # Get mapper dict for healthy TCGA samples
            with open(os.path.join(path, "mappers", "healthy_tcga_2_gtex_mapper.json"), 'r') as f:
                normal_tcga_mapper = json.load(f)
            # Modify lab_txt labels of healthy TCGA samples to corresponded GTEx labels
            tcga_meta.loc[tcga_meta['healthy'], 'lab_txt'] = tcga_meta[tcga_meta['healthy']]['tissue'].map(normal_tcga_mapper)
            

            # Process GTEx metadata #############
            # TODO: Standardize identifiers with wang and toil
            gtex_meta = gtex_meta[['Unnamed: 0','gtex.smts','gtex.smtsd']]
            # Rename columns
            gtex_meta.columns = ['sample', 'lab_gtex', 'tissue']
            # Obtain healthy column
            gtex_meta['healthy'] = gtex_meta['tissue'] != 'Cells - Leukemia cell line (CML)' # FIXME: Finally decide if this cell lines are healthy or not. For now they are considered desease
            # Reset index
            gtex_meta.set_index('sample', inplace=True)
            # Filter out rows with NaNs
            gtex_meta.dropna(inplace=True)
            # Add is_tcga column
            gtex_meta['is_tcga'] = False
            
            # Get standard lab_txt labels from gtex_meta #######
            with open(os.path.join(path, "mappers", "recount3_gtex_mapper.json"), 'r') as f:
                gtex_mapper = json.load(f)
            gtex_meta['lab_txt'] = gtex_meta['tissue'].map(gtex_mapper)

            # Merge both metadata in single dataframe ##########
            global_meta = pd.concat((gtex_meta[['lab_txt', 'is_tcga', 'healthy']], tcga_meta[['lab_txt', 'is_tcga', 'healthy']]))

            # Filter matrix data to leave only the samples with valid metadata. This line also ensures that the ordering
            # of metadata samples and matrix_data samples is the same
            matrix_data = matrix_data[global_meta.index]

            # Put gene ids in matrix_data index
            matrix_data.set_index(gene_meta['gene_id'], inplace=True)

            # Perform Log2(x+1) transformation over matrix_data
            tqdm.pandas(desc="Computing Log2(x+1) transform")
            matrix_data = matrix_data.progress_apply(lambda x: np.log2(x+1))

            # Sort samples in data matrix and global_meta
            print('Sorting samples...')
            matrix_data = matrix_data.reindex(sorted(matrix_data.columns), axis=1)
            global_meta.sort_index(inplace=True)

            # Reset index to save in feather file
            matrix_data.reset_index(inplace=True)

            print(f'Saving processed data to {os.path.join(path, "processed_data")}')
            os.makedirs(os.path.join(path, 'processed_data'), exist_ok=True)
            matrix_data.to_feather(os.path.join(path, 'processed_data', 'data_matrix.feather'))
            global_meta.to_csv(os.path.join(path, 'processed_data', 'data_category.csv'))

            # Set index again for next steps
            matrix_data.set_index('gene_id', inplace=True)
        
        else:
            print(f'Loading processed data from {os.path.join(path, "processed_data")}')
            matrix_data = pd.read_feather(os.path.join(path, 'processed_data', 'data_matrix.feather'))
            global_meta = pd.read_csv(os.path.join(path, 'processed_data', 'data_category.csv'), index_col='sample')
            gene_meta = pd.read_csv(os.path.join(path, 'original_data', "gene_metadata.csv"), low_memory=False)
            matrix_data.set_index('gene_id', inplace=True)

        # Print the time needed to read and process raw data 
        end = time.time()
        print("Time to load data: {} s".format(round(end - start, 3)))

        # Map ensemble gene id to gene symbol
        matrix_data = map_ensg_to_symbol(matrix_data)

        return matrix_data, global_meta


# Specific dataset declaration
class ToilDataset(gtex_tcga_dataset):
    def __init__(self, path, read_func=read_toil, dataset='both', tissue='all', binary_dict={}, mean_thr=-10, std_thr=0.01, rand_frac=1, sample_frac=0.5, gene_list_csv='None', wang_level=0, batch_normalization='None', fold_number=5, partition_seed=0, force_compute=False):
        super().__init__(path, read_func, dataset, tissue, binary_dict, mean_thr, std_thr, rand_frac, sample_frac, gene_list_csv, wang_level, batch_normalization, fold_number, partition_seed, force_compute)

class WangDataset(gtex_tcga_dataset):
    def __init__(self, path, read_func=read_wang, dataset='both', tissue='all', binary_dict={}, mean_thr=-10, std_thr=0.01, rand_frac=1, sample_frac=0.5, gene_list_csv='None', wang_level=0, batch_normalization='None', fold_number=5, partition_seed=0, force_compute=False):
        super().__init__(path, read_func, dataset, tissue, binary_dict, mean_thr, std_thr, rand_frac, sample_frac, gene_list_csv, wang_level, batch_normalization, fold_number, partition_seed, force_compute)

class Recount3Dataset(gtex_tcga_dataset):
    def __init__(self, path, read_func=read_recount3, dataset='both', tissue='all', binary_dict={}, mean_thr=-10, std_thr=0.01, rand_frac=1, sample_frac=0.5, gene_list_csv='None', wang_level=0, batch_normalization='None', fold_number=5, partition_seed=0, force_compute=False):
        super().__init__(path, read_func, dataset, tissue, binary_dict, mean_thr, std_thr, rand_frac, sample_frac, gene_list_csv, wang_level, batch_normalization, fold_number, partition_seed, force_compute)

# Test code for dataset declaration

# Read function tests

# toil_matrix, toil_categories            = read_toil(os.path.join('data', 'toil_data'), force_compute=False)
# wang_matrix, wang_categories            = read_wang(os.path.join('data', 'wang_data'), force_compute=False)
# recount3_matrix, recount3_categories    = read_recount3(os.path.join('data', 'recount3_data'), force_compute=False)

# Dataset tests

# test_toil =     ToilDataset(os.path.join("data", "toil_data"),          batch_normalization='normal', dataset='both', force_compute=False, sample_frac=0.5)
# test_wang =     WangDataset(os.path.join('data', 'wang_data'),          batch_normalization='normal', dataset='both', force_compute=False, sample_frac=0.99)
# test_recount3 = Recount3Dataset(os.path.join('data', 'recount3_data'),  batch_normalization='normal', dataset='both', force_compute=False, sample_frac=0.99)

# Standard Loader tests

# train_loader_toil, test_loader_toil = test_toil.get_dataloaders(batch_size = 100, fold = 0)
# train_loader_wang, test_loader_wang = test_wang.get_dataloaders(batch_size = 100, fold = 0)
# train_loader_recount3, test_loader_recount3 = test_recount3.get_dataloaders(batch_size = 100, fold = 0)

# Hong Loader tests

# hong_toil_dict =          test_toil.get_hong_dataloaders(batch_size_multitask=453,        batch_size_subtype=421, pca=2000, fold=0)
# hong_wang_dict =          test_wang.get_hong_dataloaders(batch_size_multitask=453,        batch_size_subtype=421, pca=2000, fold=0)
# hong_recount3_dict =      test_recount3.get_hong_dataloaders(batch_size_multitask=453,    batch_size_subtype=421, pca=2000, fold=0)
