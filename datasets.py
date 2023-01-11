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
import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
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


class gtex_tcga_dataset():
    def __init__(self, path, read_func, dataset = 'both', tissue='all', binary_dict={}, mean_thr=-10.0,
                std_thr=0.01, rand_frac = 1.0, sample_frac = 0.5, gene_list_csv='None',
                batch_normalization='None', fold_number=5, partition_seed=0, force_compute = False):

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
        self.batch_normalization = batch_normalization
        self.fold_number = fold_number
        self.partition_seed = partition_seed # seed for train/val/test split
        self.force_compute = force_compute

        # Main Bioinformatic pipeline
        # Make mapper files if they are not already saved
        self.make_mappers()
        # Read data from the Recount3 dataset and perform a log2(x+1) transformation. Also return gene metadata that is only available for Recount3
        self.matrix_data, self.categories = self.read_func(self.path, self.force_compute)
        # Filter Wang datasets to use GTEx, TCGA or both.
        self.matrix_data_filtered, self.categories_filtered = self.filter_datasets()
        # Get labels dataframe and label dictionary. 
        self.label_df, self.lab_txt_2_lab_num = self.find_labels()
        # Find stats of each dataset segment
        self.general_stats = self.find_general_stats()
        # Filter genes based on mean, std and sample_frac. This also subsamples the resulting filtered gene list by self.rand_frac. 
        # If self.gene_list_csv path is specified it works like a wildcard and CanDLE will train only with the genes in the csv path
        self.filtered_gene_list, self.gene_filtered_data_matrix = self.filter_genes()
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

        # FIXME: This erases the dimensionality reduction files from previous experiments
        # Make dataset directory if it does not exist
        os.makedirs(self.dataset_info_path, exist_ok = True)

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
            tcga_df = self.label_df[self.label_df['is_tcga']]

            # Get the identifiers of the samples in each subset
            gtex_samples = self.label_df[self.label_df['is_tcga']==False].index
            tcga_samples = tcga_df.index
            healthy_tcga_samples = tcga_df[tcga_df['lab_txt'].str.contains('GTEX')].index
            joint_samples = self.label_df.index

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

    # This function performs a data normalization by batches (GTEX or TCGA) 
    def batch_normalize(self):
        if self.batch_normalization=='None':
            print('Did not perform batch normalization...')
            return
        else:
            print('Batch normalizing matrix data...')
            start = time.time()
            # Get the identifiers of the samples in each subset
            gtex_samples = self.label_df[self.label_df['is_tcga']==False].index
            tcga_samples = self.label_df[self.label_df['is_tcga']==True].index
            # Get stats of the valid genes
            valid_stats = self.general_stats.loc[self.filtered_gene_list, :]

            # Transforms GTEx data
            normalized_gtex = self.gene_filtered_data_matrix[gtex_samples].sub(valid_stats['gtex_mean'], axis=0)
            normalized_gtex = normalized_gtex.div(valid_stats['gtex_std'], axis=0)
           
           # Transform TCGA data according to self.batch_normalization
            if self.batch_normalization=='normal':
                normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['tcga_mean'], axis=0)
                normalized_tcga = normalized_tcga.div(valid_stats['tcga_std'], axis=0)
            
            elif self.batch_normalization=='healthy_tcga':
                normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['healthy_tcga_mean'], axis=0)
                normalized_tcga = normalized_tcga.div(valid_stats['healthy_tcga_std'], axis=0)
            
            else:
                raise ValueError('Batch normalization should be None, normal or healthy_tcga.')

            normalized_joint = pd.concat([normalized_gtex, normalized_tcga], axis=1)

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

    # This function uses self.label_df to split the data into train, validation and test sets
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

            print('Reducing dimensions with UMAP...')
            raw_umap = UMAP(n_components=2, random_state=0) # , n_neighbors=64, local_connectivity=32)          # n_neighbors and local_connectivity are set to ensure that the graph is connected
            processed_umap = UMAP(n_components=2, random_state=0) # , n_neighbors=64, local_connectivity=32)
            r_raw_umap = raw_umap.fit_transform(raw_data)
            r_processed_umap = processed_umap.fit_transform(processed_data)

            reduced_dict = {'raw_pca': r_raw_pca,               'raw_tsne': r_raw_tsne,                 'raw_umap': r_raw_umap,
                            'processed_pca': r_processed_pca,   'processed_tsne': r_processed_tsne,     'processed_umap':  r_processed_umap}

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
            fig, ax = plt.subplots(2, 3)

            ax[0, 0].scatter(reduced_dict['raw_pca'][:,0],          reduced_dict['raw_pca'][:,1],           c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
            ax[1, 0].scatter(reduced_dict['processed_pca'][:,0],    reduced_dict['processed_pca'][:,1],     c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
            ax[0, 1].scatter(reduced_dict['raw_tsne'][:,0],         reduced_dict['raw_tsne'][:,1],          c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
            ax[1, 1].scatter(reduced_dict['processed_tsne'][:,0],   reduced_dict['processed_tsne'][:,1],    c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
            ax[0, 2].scatter(reduced_dict['raw_umap'][:,0],         reduced_dict['raw_umap'][:,1],          c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
            ax[1, 2].scatter(reduced_dict['processed_umap'][:,0],   reduced_dict['processed_umap'][:,1],    c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)

            label_dict = {0: 'PC', 1: 'T-SNE', 2: 'UMAP'}
            title_dict = {0: 'PCA', 1: 'T-SNE', 2: 'UMAP'}

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
        return matrix_data, categories

def read_wang(path, force_compute):

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
                # Exclude chol category which is said to be discarded in original paper but is in the downloaded data
                if not(class_path[:4] == 'chol'):
                #if True:
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

        # Log2(x+1) transform the data
        tqdm.pandas(desc="Computing Log2(x+1) transform")
        data_matrix = data_matrix.progress_apply(lambda x: np.log2(x+1))
        
        # Sort matrix data columns by samples name
        data_matrix = data_matrix.reindex(sorted(data_matrix.columns), axis=1)
        # Sort categories dataframe by sample name
        category_df.sort_index(inplace=True)

        # Reset index to save in feather file
        data_matrix.reset_index(inplace=True)

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
        return matrix_data, global_meta


# Specific dataset declaration
class ToilDataset(gtex_tcga_dataset):
    def __init__(self, path, read_func=read_toil, dataset='both', tissue='all', binary_dict={}, mean_thr=-10, std_thr=0.01, rand_frac=1, sample_frac=0.5, gene_list_csv='None', batch_normalization='None', fold_number=5, partition_seed=0, force_compute=False):
        super().__init__(path, read_func, dataset, tissue, binary_dict, mean_thr, std_thr, rand_frac, sample_frac, gene_list_csv, batch_normalization, fold_number, partition_seed, force_compute)

class WangDataset(gtex_tcga_dataset):
    def __init__(self, path, read_func=read_wang, dataset='both', tissue='all', binary_dict={}, mean_thr=-10, std_thr=0.01, rand_frac=1, sample_frac=0.5, gene_list_csv='None', batch_normalization='None', fold_number=5, partition_seed=0, force_compute=False):
        super().__init__(path, read_func, dataset, tissue, binary_dict, mean_thr, std_thr, rand_frac, sample_frac, gene_list_csv, batch_normalization, fold_number, partition_seed, force_compute)

class Recount3Dataset(gtex_tcga_dataset):
    def __init__(self, path, read_func=read_recount3, dataset='both', tissue='all', binary_dict={}, mean_thr=-10, std_thr=0.01, rand_frac=1, sample_frac=0.5, gene_list_csv='None', batch_normalization='None', fold_number=5, partition_seed=0, force_compute=False):
        super().__init__(path, read_func, dataset, tissue, binary_dict, mean_thr, std_thr, rand_frac, sample_frac, gene_list_csv, batch_normalization, fold_number, partition_seed, force_compute)

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

##############################################################################################################################################



# class ToilDataset():
#     def __init__(self, path, dataset = 'both', tissue='all', binary_dict={}, mean_thr=-10.0,
#                 std_thr=0.01, rand_frac = 1.0, sample_frac = 0.5, gene_list_csv='None', label_type = 'phenotype', 
#                 batch_normalization='None', partition_seed=0, force_compute = False):
#         self.path = path
#         self.tissue = tissue
#         self.binary_dict = binary_dict
#         self.dataset = dataset
#         # FIXME: Problems if we change rand_frac seed for filtering info
#         self.dataset_info_path = os.path.join(self.path, 'processed_data',
#                                               f'dataset={dataset}',
#                                               f'mean_thr={mean_thr}_std_thr={std_thr}',
#                                               f'sample_frac={sample_frac}_rand_frac={rand_frac}', 
#                                               f'tissue={self.tissue}')
#         self.mean_thr = mean_thr
#         self.std_thr = std_thr
#         self.rand_frac = rand_frac
#         self.sample_frac = sample_frac
#         self.gene_list_csv = gene_list_csv
#         self.label_type = label_type # can be 'phenotype' or 'category'
#         self.batch_normalization = batch_normalization # TODO: Just allow batch normalization under certain conditions of datasets
#         self.partition_seed = partition_seed # seed for train/val/test split
#         self.force_compute = force_compute

#         # Main Bioinformatic pipeline

#         # Make mapper files if they are not already saved
#         self.make_mappers()
#         # Read data from the Toil data set 
#         self.matrix_data, self.categories, self.phenotypes = self.read_data()
#         # Filter toil datasets to use GTEx, TCGA or both. This part also takes out TARGET data
#         self.matrix_data_filtered, self.categories_filtered, self.phenotypes_filtered = self.filter_datasets()
#         # Get labels and label dictionary. 
#         self.label_df, self.lab_txt_2_lab_num = self.find_labels()
#         # Find stats of each dataset segment
#         self.general_stats = self.find_general_stats()
#         # Filter genes based on mean, std and sample_frac. This also subsamples the resulting filtered gene list by self.rand_frac. 
#         # If self.gene_list_csv path is specified it works like a wildcard and CanDLE will train only with the genes in the csv path
#         self.filtered_gene_list, self.gene_filtered_data_matrix = self.filter_genes()
#         # TODO: Add a function that handles a possible subsampling of the data
#         # Perform batch normalization, this uses self.find_general_stats() and normalizes self.gene_filtered_data_matrix  
#         self.batch_normalize()
#         # Filter self.label_df and self.lab_txt_2_lab_num based on the specified tissue
#         self.filter_by_tissue()
#         # Make the problem binary in case self.binary_dict is not empty
#         self.make_binary_problem() # If self.binary_dict == {} this function does nothing
#         # Split data into train, validation and test sets. This function uses self.label_df to split the data with the same proportion.
#         self.split_labels, self.split_matrices = self.split_data() # For split_matrices samples are columns and genes are rows

#         # Define number of classes for classification
#         self.num_classes = len(self.lab_txt_2_lab_num.keys()) if self.binary_dict == {} else 2

#         # TODO: Make function that prints a global summary of the complete dataset (n_samples, n_genes and so on)

#         # Make important plots with dataset characteristics
#         self.plot_label_distribution() # TODO: Make a better plot with one bar per class divided in train/val/test
#         self.plot_sample_frac()
#         # self.plot_gene_expression_histograms(rand_size=100000)
#         # self.plot_dim_reduction()

#     def make_mappers(self):
#         """
#         This function generates mapper files useful for class definition in the dataset by running the make_mappers.py file
#         """
#         # Just make mappers if they are not already saved
#         if not os.path.exists(os.path.join(self.path, 'mappers', 'category_mapper.json')) or self.force_compute:
#             # run main.py with subprocess
#             command = f'python make_mappers.py'
#             print(command)
#             command = command.split()
#             subprocess.call(command)    

#     def read_data(self):
#         """
#         Reads data from the Toil data set with root path and returns matrix_data, categories and phenotypes dataframes.

#         Args:
#             path (str): Root path of the Toil data set.

#         Returns:
#             matrix_data (pd.dataframe): Matrix data of the Toil data set. Columns are samples and rows are genes.
#             categories (pd.dataframe): Categories of the Toil data set. Rows are samples.
#             phenotypes (pd.dataframe): Detailed phenotypes of the Toil data set. Rows are samples.
#         """
#         start = time.time()
#         matrix_data = pd.read_feather(os.path.join(self.path, "data_matrix.feather"))
#         categories = pd.read_csv(os.path.join(self.path, "categories.csv"), encoding = "cp1252")
#         phenotypes = pd.read_csv(os.path.join(self.path, "phenotypes.csv"), encoding = "cp1252")
#         # Delete the first column of categories and phenotypes
#         categories = categories.drop(categories.columns[0], axis = 1)
#         phenotypes = phenotypes.drop(phenotypes.columns[0], axis = 1)
#         # Set the first column of matrix_data, categories and phenotypes as index
#         matrix_data.set_index(matrix_data.columns[0], inplace = True)
#         categories.set_index(categories.columns[0], inplace = True)
#         phenotypes.set_index(phenotypes.columns[0], inplace = True)
#         # Delete the rows with nan values
#         categories.dropna(inplace=True)
#         phenotypes.dropna(inplace=True)
#         # Filter out all TARGET samples
#         matrix_data = matrix_data.iloc[:, ~matrix_data.columns.str.contains("TARGET")]
#         phenotypes = phenotypes[~phenotypes.index.str.contains("TARGET")]
#         # Add is_tcga column to categories and phenotypes
#         categories['is_tcga'] = categories['TCGA_GTEX_main_category'].str.contains('TCGA')
#         phenotypes['is_tcga'] = phenotypes['_study'] == 'TCGA'
#         end = time.time()
#         print("Time to load data: {} s".format(round(end - start, 3)))
#         return matrix_data, categories, phenotypes
    
#     def filter_datasets(self):
#         """
#         Filters the Toil data set by using or not using TCGA and GTEx samples.

#         Args:
#             matrix_data (pd.dataframe): Dataframe of the complete Toil data set obtained from read_data().
#             categories (pd.dataframe): Dataframe of categories of the complete Toil data set obtained from read_data().
#             phenotypes (pd.dataframe): Dataframe of phenotypes of the complete Toil data set obtained from read_data().
#             tcga (bool, optional): If True, TCGA samples are used. Defaults to True. 
#             gtex (bool, optional): If True GTEX samples are used. Defaults to True.

#         Raises:
#             ValueError: If both tcga and gtex are False then an error is raised because there is no data to use.

#         Returns:
#             matrix_data_filtered(pd.dataframe): Dataframe of the filtered Toil data set.
#             categories_filtered(pd.dataframe): Dataframe of the filtered categories of the Toil data set.
#             phenotypes_filtered(pd.dataframe): Dataframe of the filtered phenotypes of the Toil data set.
#         """

#         # Handle the filters for TCGA and GTEx
#         if self.dataset == 'tcga':
#             print("Using TCGA samples only")
#             # Filter out all gtex samples from matrix_data
#             matrix_data_filtered = self.matrix_data.iloc[:, ~self.matrix_data.columns.str.contains("GTEX")]
#             categories_filtered = self.categories.loc[~self.categories.index.str.contains("GTEX"), :]
#             phenotypes_filtered = self.phenotypes.loc[~(self.phenotypes["_study"]=="GTEX"), :]
#         elif self.dataset == 'gtex':
#             print("Using GTEX samples only")
#             # Filter out all tcga samples from matrix_data
#             matrix_data_filtered = self.matrix_data.iloc[:, ~self.matrix_data.columns.str.contains("TCGA")]
#             categories_filtered = self.categories.loc[~self.categories.index.str.contains("TCGA"), :]
#             phenotypes_filtered = self.phenotypes.loc[~(self.phenotypes["_study"]=="TCGA"), :]
#         elif self.dataset == 'both':
#             # Do nothing because both TCGA and GTEX samples are included
#             print("Using TCGA and GTEX samples")
#             matrix_data_filtered = self.matrix_data
#             categories_filtered = self.categories
#             phenotypes_filtered = self.phenotypes
            
#         return matrix_data_filtered, categories_filtered, phenotypes_filtered

#     # This function extracts the labels from categories_filtered or phenotypes_filtered and returns a label dataframe and a dictionary of textual labels to numeric labels
#     def find_labels(self):
#         # Load mapper dict from normal TCGA samples to GTEX category
#         with open(os.path.join(self.path, "mappers", "normal_tcga_2_gtex_mapper.json"), "r") as f:
#             normal_tcga_2_gtex = json.load(f)

#         # Make dataset directory if it does not exist
#         os.makedirs(self.dataset_info_path, exist_ok = True)

#         if self.label_type == 'category':
#             label_df = self.categories_filtered
            
#             # Load the categories to textual labels dictionary.
#             with open(os.path.join(self.path, "mappers", "category_mapper.json"), "r") as f:
#                 cat_2_lab_txt = json.load(f)

#             # Note: The self.categories_filtered dataframe does not contain any normal (Healthy) TCGA samples.
#             # Add one column to the label_df with mapping from TCGA_GTEX_main_category column with cat_2_lab_txt dictionary
#             label_df["lab_txt"] = label_df["TCGA_GTEX_main_category"].map(cat_2_lab_txt)
#             # Get unique textual labels obtained and sort them
#             current_labels = sorted(label_df["lab_txt"].unique().tolist())
#             # Define lab_txt_2_lab_num dictionary
#             lab_txt_2_lab_num = {lab_txt: i for i, lab_txt in enumerate(current_labels)}

#             # Define numeric labels from the textual labels in label_df
#             label_df["lab_num"] = label_df["lab_txt"].map(lab_txt_2_lab_num)

#         elif self.label_type == 'phenotype':
#             label_df = self.phenotypes_filtered

#             # Load the phenotypes to textual labels dictionary.
#             with open(os.path.join(self.path, "mappers", "phenotype_mapper.json"), "r") as f:
#                 pheno_2_lab_txt = json.load(f)

#             # Declare a new empty column in the label_df for textual labels
#             label_df["lab_txt"] = 0
#             # Find sample names of normal (Healthy) TCGA subjects
#             normal_tcga_samples = self.phenotypes_filtered[self.phenotypes_filtered["_sample_type"] == "Solid Tissue Normal"].index
            
#             # Handle normal (Healthy) TCGA subjects
#             # If there are both TCGA and GTEX assign GTEX textual label to the normal (Healthy) TCGA subjects
#             if self.dataset == 'both':
#                 # Put GTEX textual label in lab_txt column for normal (Healthy) TCGA samples
#                 label_df.loc[normal_tcga_samples, "lab_txt"] = label_df.loc[normal_tcga_samples, "_primary_site"].map(normal_tcga_2_gtex)
#                 # pass
#             # If there is only TCGA assign TCGA-NT label to the normal (Healthy) TCGA subjects
#             elif self.dataset == 'tcga':
#                 # Put TCGA textual label in lab_txt column for normal (Healthy) TCGA samples
#                 label_df.loc[normal_tcga_samples, "lab_txt"] = "TCGA-NT"
#             else:
#                 pass

#             # Map phenotype detailed category to textual label in label_df for the non normal (Healthy) TCGA samples
#             label_df.loc[label_df["lab_txt"] == 0, "lab_txt"] = label_df.loc[label_df["lab_txt"] == 0, "detailed_category"].map(pheno_2_lab_txt)
#             # Get unique textual labels obtained and sort them
#             current_labels = sorted(label_df["lab_txt"].unique().tolist())
#             # Define lab_txt_2_lab_num dictionary
#             lab_txt_2_lab_num = {lab_txt: i for i, lab_txt in enumerate(current_labels)}
#             # Define numeric labels from the textual labels in label_df
#             label_df["lab_num"] = label_df["lab_txt"].map(lab_txt_2_lab_num)
#         else:
#             raise ValueError("label_type must be 'category' or 'phenotype'")
        
#         # Save lab_txt_2_lab_num dictionary to json file
#         with open(os.path.join(self.dataset_info_path, "lab_txt_2_lab_num_mapper.json"), "w") as f:
#             json.dump(lab_txt_2_lab_num, f, indent = 4)
#         return label_df, lab_txt_2_lab_num
    
#     # This function finds the mean expression, std and expressed sample fraction for GTEx, TCGA, healthy TCGA and the joint dataset
#     def find_general_stats(self):
#         # If the info stats are already computed load them from file
#         if (os.path.exists(os.path.join(self.path, 'general_stats.csv'))) & (self.force_compute == False):
#             print('Loading general stats from '+os.path.join(self.path, 'general_stats.csv'))
#             general_stats = pd.read_csv(os.path.join(self.path, 'general_stats.csv'), index_col = 0)
#         # If the stats are not computed compute them and save them in file
#         else:
#             print('Computing general stats and saving to '+os.path.join(self.path, 'general_stats.csv'))
#             # Define auxiliary tcga dataframe to obtain healthy tcga samples
#             tcga_df = self.label_df[self.label_df['_study']=='TCGA']

#             # Get the identifiers of the samples in each subset
#             gtex_samples = self.label_df[self.label_df['_study']=='GTEX'].index
#             tcga_samples = tcga_df.index
#             healthy_tcga_samples = tcga_df[tcga_df['lab_txt'].str.contains('GTEX')].index
#             joint_samples = self.label_df.index

#             # Compute the mean of the subsets
#             tqdm.pandas(desc="Computing Mean GTEx")
#             gtex_mean = self.matrix_data_filtered.loc[:, gtex_samples].progress_apply(np.mean, axis = 1).to_frame(name='gtex_mean')
#             tqdm.pandas(desc="Computing Mean TCGA")
#             tcga_mean = self.matrix_data_filtered.loc[:, tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='tcga_mean')
#             tqdm.pandas(desc="Computing Mean Healthy TCGA")
#             healthy_tcga_mean = self.matrix_data_filtered.loc[:, healthy_tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='healthy_tcga_mean')
#             tqdm.pandas(desc="Computing Joint Mean")
#             joint_mean = self.matrix_data_filtered.loc[:, joint_samples].progress_apply(np.mean, axis = 1).to_frame(name='joint_mean')

#             # Compute the std of the subsets
#             tqdm.pandas(desc="Computing std GTEx")
#             gtex_std = self.matrix_data_filtered.loc[:, gtex_samples].progress_apply(np.std, axis = 1).to_frame(name='gtex_std')
#             tqdm.pandas(desc="Computing std TCGA")
#             tcga_std = self.matrix_data_filtered.loc[:, tcga_samples].progress_apply(np.std, axis = 1).to_frame(name='tcga_std')
#             tqdm.pandas(desc="Computing std Healthy TCGA")
#             healthy_tcga_std = self.matrix_data_filtered.loc[:, healthy_tcga_samples].progress_apply(np.std, axis = 1).to_frame(name='healthy_tcga_std')
#             tqdm.pandas(desc="Computing Joint std")
#             joint_std = self.matrix_data_filtered.loc[:, joint_samples].progress_apply(np.std, axis = 1).to_frame(name='joint_std')

#             # Compute the fraction of samples where a gene is expressed
#             print('Computing fraction of samples where each gene is expressed ...')
#             min_val = self.matrix_data_filtered.min().min() # Get minimum value
#             tqdm.pandas(desc="Computing Expressed Genes")
#             expressed_matrix = self.matrix_data_filtered.progress_apply(lambda x: x>min_val, axis = 1) # Compute expressed positions
            
#             tqdm.pandas(desc="Computing Joint Expressed Sample Fraction")
#             joint_sample_fraction = expressed_matrix.progress_apply(np.mean, axis = 1).to_frame(name='joint_sample_frac')
#             tqdm.pandas(desc="Computing GTEx Expressed Sample Fraction")
#             gtex_sample_fraction = expressed_matrix.loc[:, gtex_samples].progress_apply(np.mean, axis = 1).to_frame(name='gtex_sample_frac')
#             tqdm.pandas(desc="Computing TCGA Expressed Sample Fraction")
#             tcga_sample_fraction = expressed_matrix.loc[:, tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='tcga_sample_frac')
            

#             # Join stats in single dataframe
#             general_stats = pd.concat([gtex_mean, tcga_mean, healthy_tcga_mean, joint_mean,
#                                         gtex_std, tcga_std, healthy_tcga_std, joint_std,
#                                         joint_sample_fraction, gtex_sample_fraction, tcga_sample_fraction,], axis=1)
#             general_stats.to_csv(os.path.join(self.path, 'general_stats.csv'))

#         return general_stats

#     # This function filters out genes by mean, standard deviation, expression fraction, random fraction or list of genes
#     def filter_genes(self):
#         # If there is a gene list specified by parameter then it overwrites mean, std and rand_frac filtering  
#         if self.gene_list_csv != 'None':
#             # Print user message
#             print(f'CanDLE will train with the list of genes specified in {self.gene_list_csv}')
#             gene_csv_df = pd.read_csv(self.gene_list_csv, index_col=0)
#             gene_list = pd.Index(gene_csv_df['gene_name'])
        
#         # If no list of genes is specified then proceed with mean, std, sample_frac and rand_frac filtering
#         elif (not os.path.exists(os.path.join(self.dataset_info_path, "filtering_info.csv"))) or self.force_compute:
            
#             print("Computing list of filtered genes. And saving filtering info to:\n\t"+ os.path.join(self.dataset_info_path, "filtering_info.csv"))
            
#             # Find the indices of the samples with mean, standard deviation and sample fractions that fulfill the thresholds
#             mean_bool_index = ((self.general_stats['joint_mean']>self.mean_thr) & (self.general_stats['gtex_mean']>self.mean_thr) & (self.general_stats['tcga_mean']>self.mean_thr))
#             std_bool_index = ((self.general_stats['joint_std']>self.std_thr) & (self.general_stats['gtex_std']>self.std_thr) & (self.general_stats['tcga_std']>self.std_thr))
#             sample_frac_bool_index = ((self.general_stats['joint_sample_frac'] > self.sample_frac) & (self.general_stats['gtex_sample_frac'] > self.sample_frac) & (self.general_stats['tcga_sample_frac'] > self.sample_frac))
            
#             # Compute intersection of mean_data_index and std_data_index
#             mean_std_sample_index = np.logical_and.reduce((mean_bool_index.values, std_bool_index.values, sample_frac_bool_index)).ravel()
#             # Make a gene list of the samples that fulfill the thresholds
#             gene_list = self.matrix_data.index[mean_std_sample_index]

#             # Subsample gene list in case self.rand_frac < 1
#             if self.rand_frac < 1:
#                 np.random.seed(0) # Ensure reproducibility # TODO: Parametrize this seed to run variation experiments
#                 rand_selector = np.zeros(len(gene_list))
#                 rand_selector[:int(len(gene_list)*self.rand_frac)] = 1
#                 np.random.shuffle(rand_selector) # Shuffle boolean selector
#                 rand_selector = np.array(rand_selector, dtype=bool)
#                 gene_list = gene_list[rand_selector] # Filter gene list based in rand_selector
            
#             # Compute boolean value for each gene that indicates if it was included in the filtered gene list
#             included_in_filtering = self.general_stats.index.isin(gene_list)

#             # Merge all statistics and included_in_filtering into a final dataframe
#             filtering_info_df = self.general_stats
#             filtering_info_df['included'] = included_in_filtering
#             filtering_info_df.index.name = "gene"

#             # Save the gene list, mean and standard deviation of the matrix_data to files
#             filtering_info_df.to_csv(os.path.join(self.dataset_info_path, "filtering_info.csv"), index = True)
#             # Plot histograms with plot_filtering_histograms()
#             self.plot_filtering_histograms(filtering_info_df)

#         else:
#             print("Loading filtering info from:\n\t" + os.path.join(self.dataset_info_path, "filtering_info.csv"))
#             filtering_info_df = pd.read_csv(os.path.join(self.dataset_info_path, "filtering_info.csv"), index_col = 0)
#             # get indices of filtering_info_df that are True in the included column
#             gene_list = filtering_info_df.index[filtering_info_df["included"].values == True]
#             # Plot histograms with plot_filtering_histograms()
#             self.plot_filtering_histograms(filtering_info_df)
        
#         # Filter tha data matrix based on the gene list
#         gene_filtered_data_matrix = self.matrix_data_filtered.loc[gene_list, :]

#         print("Currently working with {} genes...".format(gene_filtered_data_matrix.shape[0]))

#         return gene_list.to_list(), gene_filtered_data_matrix
    
#     # This function plots a 2X2 figure with the histograms of mean expression and standard deviation before and after filtering
#     def plot_filtering_histograms(self, filtering_info_df):        
#         # Make a figure
#         fig, axes = plt.subplots(2, 2, figsize = (18, 12))
#         fig.suptitle("Filtering of Mean > " +str(self.mean_thr) + " and Standard Deviation > " + str(self.std_thr) , fontsize = 30)
#         # Variable to adjust display height of the histograms
#         max_hist = np.zeros((2, 2))

#         # Plot the histograms of mean and standard deviation before filtering
#         n, _, _ = axes[0, 0].hist(filtering_info_df["joint_mean"], bins = 50, color = "k", density=True)
#         max_hist[0, 0] = np.max(n)
#         axes[0, 0].set_title("Before filtering", fontsize = 26)        
#         n, _, _ = axes[1, 0].hist(filtering_info_df["joint_std"], bins = 50, color = "k", density=True)
#         max_hist[1, 0] = np.max(n)
#         # Plot the histograms of mean and standard deviation after filtering
#         n, _, _ = axes[0, 1].hist(filtering_info_df["joint_mean"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
#         max_hist[0, 1] = np.max(n)
#         axes[0, 1].set_title("After filtering", fontsize = 26)
#         n, _, _ = axes[1, 1].hist(filtering_info_df["joint_std"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
#         max_hist[1, 1] = np.max(n)

#         # Format axes
#         for i in range(2):
#             for j in range(2):
#                 axes[i, j].set_ylabel("Density", fontsize = 16)
#                 axes[i, j].tick_params(labelsize = 14)
#                 axes[i, j].grid(True)
#                 axes[i, j].set_axisbelow(True)
#                 axes[i, j].set_ylim(0, max_hist[i, j] * 1.1)
#                 # Handle mean expression plots
#                 if i == 0:
#                     axes[i, j].set_xlabel("Mean expression", fontsize = 16)
#                     axes[i, j].set_xlim(filtering_info_df["joint_mean"].min(), filtering_info_df["joint_mean"].max())
#                     axes[i, j].plot([self.mean_thr, self.mean_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")
#                 # Handle standard deviation plots
#                 else:
#                     axes[i, j].set_xlabel("Standard deviation", fontsize = 16)
#                     axes[i, j].set_xlim(filtering_info_df["joint_std"].min(), filtering_info_df["joint_std"].max())
#                     axes[i, j].plot([self.std_thr, self.std_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")

#         # Save the figure
#         fig.savefig(os.path.join(self.dataset_info_path, "filtering_histograms.png"), dpi = 300)
#         plt.close(fig)

#     # This function performs a data normalization 
#     def batch_normalize(self):
#         if self.batch_normalization=='None':
#             print('Did not perform batch normalization...')
#             return
#         else:
#             print('Batch normalizing matrix data...')
#             start = time.time()
#             # Define auxiliary tcga dataframe to obtain healthy tcga samples
#             tcga_df = self.label_df[self.label_df['_study']=='TCGA']
#             # Get the identifiers of the samples in each subset
#             gtex_samples = self.label_df[self.label_df['_study']=='GTEX'].index
#             tcga_samples = tcga_df.index
#             # Get stats of the valid genes
#             valid_stats = self.general_stats.loc[self.filtered_gene_list, :]

#             # Transforms GTEx data
#             normalized_gtex = self.gene_filtered_data_matrix[gtex_samples].sub(valid_stats['gtex_mean'], axis=0)
#             normalized_gtex = normalized_gtex.div(valid_stats['gtex_std'], axis=0)
           
#            # Transform TCGA data according to self.batch_normalization
#             if self.batch_normalization=='normal':
#                 normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['tcga_mean'], axis=0)
#                 normalized_tcga = normalized_tcga.div(valid_stats['tcga_std'], axis=0)
            
#             elif self.batch_normalization=='healthy_tcga':
#                 normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['healthy_tcga_mean'], axis=0)
#                 normalized_tcga = normalized_tcga.div(valid_stats['healthy_tcga_std'], axis=0)
            
#             else:
#                 raise ValueError('Batch normalization should be None, normal or healthy_tcga.')

#             normalized_joint = pd.concat([normalized_gtex, normalized_tcga], axis=1)
#             # Replace NaNs generated by std division to 0's
#             self.gene_filtered_data_matrix = normalized_joint.fillna(0.0)
#             end = time.time()
#             print(f'It took {round(end-start, 2)} s to batch normalize the data.')
        
#     # This function modifies self.label_df and self.lab_txt_2_lab_num filtering by the specified tissue in self.tissue
#     def filter_by_tissue(self):
#         # If tissue is not specified, do not filter
#         if self.tissue == 'all':
#             print("No Filtering by tissue using all samples.")
#             print("Number of samples after filtering by tissue: {}".format(len(self.label_df)))
#             return
#         # If tissue is specified, filter label_df and lab_txt_2_lab_num
#         else:
#             # Load id_2_tissue mapper from file
#             with open(os.path.join(self.path, "mappers", "id_2_tissue_mapper.json"), "r") as f:
#                 id_2_tissue_mapper = json.load(f)
#             # Handle tha case where tissue is not in mapper
#             if self.tissue not in id_2_tissue_mapper.values():
#                 raise ValueError("Tissue {} is not in the tissue mapper.".format(self.tissue))
#             # Define new column in label_df with tissue labels
#             self.label_df["tissue"] = self.label_df["lab_txt"].map(id_2_tissue_mapper)
#             # Filter label_df by tissue
#             self.label_df = self.label_df[self.label_df["tissue"] == self.tissue]
#             # Re define current labels
#             current_labels = sorted(self.label_df["lab_txt"].unique().tolist())
#             # Re compute lab_txt_2_lab_num dictionary
#             self.lab_txt_2_lab_num = {lab_txt: i for i, lab_txt in enumerate(current_labels)}
#             # Define numeric labels from the textual labels in self.label_df
#             self.label_df["lab_num"] = self.label_df["lab_txt"].map(self.lab_txt_2_lab_num)
        
#         print("Filtered by {} tissue".format(self.tissue))
#         print("Number of samples after filtering by tissue: {}".format(len(self.label_df)))

#     # This function uses self.binary_dict to modify self.label_df and self.lab_txt_2_lab_num to make the labels binary
#     def make_binary_problem(self):
#         # If binary_dict is not specified, do not make binary
#         if self.binary_dict == {}:
#             print("No binary problem specified.")
#             return
#         # If binary_dict is specified, make binary
#         else:
#             self.lab_txt_2_lab_num = self.binary_dict
#             # Define numeric labels from the textual labels in self.label_df
#             self.label_df["lab_num"] = self.label_df["lab_txt"].map(self.lab_txt_2_lab_num)   
#             print("Made binary problem.")
#             print("Number of samples in class 0: {}, number of samples in class 1: {}".format(len(self.label_df[self.label_df["lab_num"] == 0]), len(self.label_df[self.label_df["lab_num"] == 1])))
#             return

#     # This function uses self.label_df to split the data into train, validation and test sets
#     def split_data(self):
#         train_val_lab, test_lab = train_test_split(self.label_df["lab_num"], test_size = 0.2, random_state = self.partition_seed, stratify = self.label_df["lab_num"].values)
#         train_lab, val_lab = train_test_split(train_val_lab, test_size = 0.25, random_state = self.partition_seed, stratify = train_val_lab.values)
#         # Use label indexes to subset the data in self.matrix_data_filtered
#         train_matrix = self.gene_filtered_data_matrix[train_lab.index]
#         val_matrix = self.gene_filtered_data_matrix[val_lab.index]
#         test_matrix = self.gene_filtered_data_matrix[test_lab.index]
#         train_val_matrix = self.gene_filtered_data_matrix[train_val_lab.index]
#         # Declare label dictionaries
#         split_labels = {"train": train_lab, "val": val_lab, "test": test_lab, "train_val": train_val_lab}
#         # Declare matrix dictionaries
#         split_matrices = {"train": train_matrix, "val": val_matrix, "test": test_matrix, "train_val": train_val_matrix}
#         # Both matrixes and labels are already shuffled
#         return split_labels, split_matrices

#     # This function gets the dataloaders for the train, val and test sets
#     def get_dataloaders(self, batch_size):
#         # Select data partitions
#         # These data matrices have samples in rows and genes in columns
#         x_train = torch.Tensor(self.split_matrices["train"].T.values).type(torch.float)
#         x_val = torch.Tensor(self.split_matrices["val"].T.values).type(torch.float)
#         x_test = torch.Tensor(self.split_matrices["test"].T.values).type(torch.float)
        
#         # Cast labels as tensors
#         y_train = torch.Tensor(self.split_labels["train"].values).type(torch.long)
#         y_val = torch.Tensor(self.split_labels["val"].values).type(torch.long)
#         y_test = torch.Tensor(self.split_labels["test"].values).type(torch.long)

#         # Define train, val and test datasets
#         train_dataset = TensorDataset(x_train, y_train)
#         val_dataset = TensorDataset(x_val, y_val)
#         test_dataset = TensorDataset(x_test, y_test)

#         # Create dataloaders
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#         return train_loader, val_loader, test_loader
    
#     # This function plots the label distribution of the dataset
#     def plot_label_distribution(self):
#         # Reverse self.lab_txt_2_lab_num dictionary
#         lab_num_2_lab_txt = {v: k for k, v in self.lab_txt_2_lab_num.items()}

#         # Get label distribution
#         train_label_dist = self.split_labels["train"].value_counts()
#         val_label_dist = self.split_labels["val"].value_counts()
#         test_label_dist = self.split_labels["test"].value_counts()

#         # Give distribution textual label indexes
#         train_label_dist.index = train_label_dist.index.map(lab_num_2_lab_txt)
#         val_label_dist.index = val_label_dist.index.map(lab_num_2_lab_txt)
#         test_label_dist.index = test_label_dist.index.map(lab_num_2_lab_txt)
        
#         # Handle different fig sizes for gtex and tcga
#         fig_size = (15, 15) if self.dataset == 'both' else (15, 7)

#         # Plot horizontal bar chart of label distribution
#         plt.figure(figsize=fig_size)
#         ax = plt.subplot(1, 3, 1)
#         train_label_dist.plot(kind="barh", color="blue", alpha=0.7)
#         plt.title("Train")
#         plt.xlabel("Label count")
#         plt.ylabel("Label")
#         plt.yticks(fontsize=8)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)

#         ax = plt.subplot(1, 3, 2)
#         val_label_dist.plot(kind="barh", color="green", alpha=0.7)
#         plt.title("Validation")
#         plt.xlabel("Label count")
#         plt.ylabel("Label")
#         plt.yticks(fontsize=8)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)

#         ax = plt.subplot(1, 3, 3)
#         test_label_dist.plot(kind="barh", color="red", alpha=0.7)
#         plt.title("Test")
#         plt.xlabel("Label count")
#         plt.ylabel("Label")
#         plt.yticks(fontsize=8)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)

#         plt.tight_layout()

#         plt.show()
#         plt.savefig(os.path.join(self.dataset_info_path, "label_distribution.png"), dpi=300)
#         plt.close()
    
#     # This function makes histograms of the gene expression values for each dataset Gtex and TCGA
#     def plot_gene_expression_histograms(self, rand_size=10000):

#         # Get just GTEx  and just TCGA matrices
#         gtex_matrix = self.matrix_data.iloc[:, self.matrix_data.columns.str.contains("GTEX")]
#         tcga_matrix = self.matrix_data.iloc[:, self.matrix_data.columns.str.contains("TCGA")]

#         print("Started sampling gene expression values...")
#         np.random.seed(0)
#         start = time.time()
#         gtex_random_sample = np.random.choice(np.ravel(gtex_matrix.values), size=rand_size)
#         tcga_random_sample = np.random.choice(np.ravel(tcga_matrix.values), size=rand_size)
#         end = time.time()
#         print("Time to sample: {}".format(round(end - start, 3)))
#         plt.figure(figsize=(9, 7))
#         # Get gene expression histograms
#         plt.hist(gtex_random_sample, bins=100, color="blue", alpha=0.7, label="GTEx", log=True, density=True)
#         plt.hist(tcga_random_sample, bins=100, color="red", alpha=0.7, label="TCGA", log=True, density=True)
#         plt.grid()
#         plt.gca().set_axisbelow(True)
#         plt.xlabel("Gene Expression $[\log_2(TPM+0.001)]$", fontsize=18)
#         plt.ylabel("Density", fontsize=18)
#         plt.xticks(fontsize=12)
#         plt.yticks(fontsize=12)
#         plt.title("Gene Expression Histograms ("+str(rand_size)+" samples)", fontsize=20)
#         plt.legend(["GTEx", "TCGA"], fontsize=14)
#         plt.tight_layout()
#         plt.show()
#         plt.savefig(os.path.join(self.dataset_info_path, "gene_expression_histograms.png"), dpi=300)
#         plt.close()

#     # This function plots an histogram of the fraction of samples that express each gene
#     def plot_sample_frac(self):
#         # Obtain minimum of filtering
#         self.general_stats['min_sample_frac'] = self.general_stats[['joint_sample_frac', 'gtex_sample_frac','tcga_sample_frac']].min(axis=1)
#         var_list = ['joint_sample_frac', 'gtex_sample_frac', 'tcga_sample_frac', 'min_sample_frac']
#         tit_list = ['Joint Fraction of Samples Where\nGene is Expressed', 'GTEx Fraction of Samples Where\nGene is Expressed',
#                     'TCGA Fraction of Samples Where\nGene is Expressed', 'Min(GTEx/TCGA) Fraction of Samples Where\nGene is Expressed']
#         color_list = ['dodgerblue', 'k', 'darkcyan', 'darkseagreen']
#         # Make a figure
#         fig, axes = plt.subplots(2, 2, figsize = (18, 12))
#         for i in range(4):
#             curr_row = i//2
#             curr_col = i%2
#             curr_ax = axes[curr_row, curr_col]
#             self.general_stats[var_list[i]].hist(bins=100, grid=False, color=color_list[i], ax=curr_ax)
#             curr_ax.set_title(tit_list[i])
#             curr_ax.set_ylabel('Frequency')
#             curr_ax.set_xlabel('Sample Fraction')
#             curr_ax.set_xlim([0,1])
#             curr_ax.spines['top'].set_visible(False)
#             curr_ax.spines['right'].set_visible(False)
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.path, 'expressed_sample_fraction_hist.png'))
#         plt.close()          

#     def plot_dim_reduction(self):
        
#         valid_samples = self.gene_filtered_data_matrix.columns
#         valid_genes = self.filtered_gene_list
#         bool_sample_index = self.matrix_data.columns.isin(valid_samples)
#         bool_gene_index = self.matrix_data.index.isin(valid_genes)
#         print('Filtering raw data to valid genes and samples...')
#         raw_data = self.matrix_data.loc[bool_gene_index, bool_sample_index].T.sort_index()
#         processed_data = self.gene_filtered_data_matrix.T.sort_index()
#         # Get and sort metadata
#         meta_df = self.label_df
#         meta_df = meta_df.sort_index()

#         if (not os.path.exists(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'))) or self.force_compute:
            
#             print(f'Computing dimensionality reduction and saving it to {os.path.join(self.dataset_info_path, "dimensionality_reduction.pkl")}...')
            
#             print('Reducing dimensions with PCA...')
#             raw_pca = PCA(n_components=2, random_state=0)
#             processed_pca = PCA(n_components=2, random_state=0)
#             r_raw_pca = raw_pca.fit_transform(raw_data)
#             r_processed_pca = processed_pca.fit_transform(processed_data)

#             # If this shows a warning with OpenBlast you can solve it using this GitHub issue https://github.com/ultralytics/yolov5/issues/2863
#             # You just need to write in terminal "export OMP_NUM_THREADS=1"
#             print('Reducing dimensions with TSNE...')
#             raw_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='random', n_jobs=-1, verbose=2)       # learning_rate and init were set due to a future warning
#             processed_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='random', n_jobs=-1, verbose=2)
#             r_raw_tsne = raw_tsne.fit_transform(raw_data)
#             r_processed_tsne = processed_tsne.fit_transform(processed_data)

#             print('Reducing dimensions with UMAP...')
#             raw_umap = UMAP(n_components=2, random_state=0) # , n_neighbors=64, local_connectivity=32)          # n_neighbors and local_connectivity are set to ensure that the graph is connected
#             processed_umap = UMAP(n_components=2, random_state=0) # , n_neighbors=64, local_connectivity=32)
#             r_raw_umap = raw_umap.fit_transform(raw_data)
#             r_processed_umap = processed_umap.fit_transform(processed_data)

#             reduced_dict = {'raw_pca': r_raw_pca,               'raw_tsne': r_raw_tsne,                 'raw_umap': r_raw_umap,
#                             'processed_pca': r_processed_pca,   'processed_tsne': r_processed_tsne,     'processed_umap':  r_processed_umap}

#             with open(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'), 'wb') as f:
#                 pkl.dump(reduced_dict, f)
        
#         else:
            
#             print(f'Loading dimensionality reduction from {os.path.join(self.dataset_info_path, "dimensionality_reduction.pkl")}...')
            
#             with open(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'), 'rb') as f:
#                 reduced_dict = pkl.load(f)
        

#         def plot_dim_reduction(reduced_dict, meta_df, color_type='tcga', cmap=None):


#             # Load id_2_tissue mapper from file
#             with open(os.path.join(self.path, "mappers", "id_2_tissue_mapper.json"), "r") as f:
#                 id_2_tissue_mapper = json.load(f)

#             meta_df['tissue'] = meta_df['lab_txt'].map(id_2_tissue_mapper)

#             # Get dictionaries to have different options to colorize the scatter points
#             tcga_dict = {True: 1, False: 0}
#             tissue_dict = {tissue: i/len(meta_df.tissue.unique()) for i, tissue in enumerate(sorted(meta_df.tissue.unique()))}
#             class_dict = {cl: i/len(meta_df.lab_txt.unique()) for i, cl in enumerate(sorted(meta_df.lab_txt.unique()))}

#             # Define color map
#             if cmap is None:
#                 d_colors = ["black", "darkcyan"]
#                 cmap = LinearSegmentedColormap.from_list("candle_cmap", d_colors)
#             else:
#                 cmap = get_cmap(cmap)
            
#             # Compute color values
#             if color_type == 'tcga':
#                 meta_df['color'] = meta_df['is_tcga'].map(tcga_dict)
#             elif color_type == 'tissue':
#                 meta_df['color'] = meta_df['tissue'].map(tissue_dict)
#             elif color_type == 'class':
#                 meta_df['color'] = meta_df['lab_txt'].map(class_dict)

#             # Plot figure
#             fig, ax = plt.subplots(2, 3)

#             ax[0, 0].scatter(reduced_dict['raw_pca'][:,0],          reduced_dict['raw_pca'][:,1],           c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[1, 0].scatter(reduced_dict['processed_pca'][:,0],    reduced_dict['processed_pca'][:,1],     c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[0, 1].scatter(reduced_dict['raw_tsne'][:,0],         reduced_dict['raw_tsne'][:,1],          c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[1, 1].scatter(reduced_dict['processed_tsne'][:,0],   reduced_dict['processed_tsne'][:,1],    c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[0, 2].scatter(reduced_dict['raw_umap'][:,0],         reduced_dict['raw_umap'][:,1],          c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[1, 2].scatter(reduced_dict['processed_umap'][:,0],   reduced_dict['processed_umap'][:,1],    c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)

#             label_dict = {0: 'PC', 1: 'T-SNE', 2: 'UMAP'}
#             title_dict = {0: 'PCA', 1: 'T-SNE', 2: 'UMAP'}

#             for i in range(ax.shape[0]):
#                 for j in range(ax.shape[1]):
#                     ax[i,j].spines.right.set_visible(False) 
#                     ax[i,j].spines.top.set_visible(False)
#                     ax[i,j].set_xlabel(f'{label_dict[j]}1')
#                     ax[i,j].set_ylabel(f'{label_dict[j]}2')
#                     ax[i,j].set_title(f'{title_dict[j]} of Raw Data' if i==0 else f'{title_dict[j]} of Processed Data')

#             # Set the legend
#             if (color_type == 'tcga'):
#                 handles = [mpatches.Patch(facecolor=cmap(0.0), edgecolor='black', label='GTEx'),
#                            mpatches.Patch(facecolor=cmap(1.0), edgecolor='black', label='TCGA')]
#                 fig.set_size_inches(13, 7)
#                 ax[0, 2].legend(handles=handles, loc='center left',bbox_to_anchor=(1.15, 0.49))
#                 plt.tight_layout()

#             elif (color_type == 'tissue'):
#                 handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'{key} Tissue') for key, val in tissue_dict.items()]
#                 fig.set_size_inches(15, 7)
#                 fig.legend(handles=handles, loc=7)
#                 fig.tight_layout()
#                 fig.subplots_adjust(right=0.83)
                

#             elif (color_type == 'class'):
#                 handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'{key}') for key, val in class_dict.items()]
#                 fig.set_size_inches(18, 8)
#                 fig.legend(handles=handles, loc=7, ncol=2)
#                 fig.tight_layout()
#                 fig.subplots_adjust(right=0.75)

            
#             else:
#                 raise ValueError('Invalid color type...')

#             # Save
#             fig.savefig(os.path.join(self.dataset_info_path, f'dim_reduction_{color_type}.png'), dpi=300)
#             plt.close()

#         # Plot dimensionality reduction with the three different color styles
#         plot_dim_reduction(reduced_dict, meta_df, color_type='tcga')
#         plot_dim_reduction(reduced_dict, meta_df, color_type='tissue', cmap='brg')
#         plot_dim_reduction(reduced_dict, meta_df, color_type='class', cmap='brg')



# class WangDataset():
#     def __init__(self, path, dataset = 'both', tissue='all', binary_dict={}, mean_thr=-10.0,
#                 std_thr=0.01, rand_frac = 1.0, sample_frac = 0.5, gene_list_csv='None',
#                 batch_normalization='None', partition_seed=0, force_compute = False):

#         self.path = path
#         self.tissue = tissue
#         self.binary_dict = binary_dict
#         self.dataset = dataset
#         # FIXME: Problems if we change rand_frac seed for filtering info
#         self.dataset_info_path = os.path.join(self.path, 'processed_data',
#                                               f'dataset={dataset}',
#                                               f'mean_thr={mean_thr}_std_thr={std_thr}',
#                                               f'sample_frac={sample_frac}_rand_frac={rand_frac}', 
#                                               f'tissue={self.tissue}')
#         self.mean_thr = mean_thr
#         self.std_thr = std_thr
#         self.rand_frac = rand_frac
#         self.sample_frac = sample_frac
#         self.gene_list_csv = gene_list_csv
#         self.batch_normalization = batch_normalization
#         self.partition_seed = partition_seed # seed for train/val/test split
#         self.force_compute = force_compute

#         # Main Bioinformatic pipeline
#         # Make mapper files if they are not already saved
#         self.make_mappers()
#         # Un-compress data
#         self.unzip_data()
#         # Read data from the Wang data set and performs a log2(x+1) transformation
#         self.matrix_data, self.categories = self.read_data()
#         # Filter Wang datasets to use GTEx, TCGA or both.
#         self.matrix_data_filtered, self.categories_filtered = self.filter_datasets()
#         # Get labels dataframe and label dictionary. 
#         self.label_df, self.lab_txt_2_lab_num = self.find_labels()
#         # Find stats of each dataset segment
#         self.general_stats = self.find_general_stats()
#         # Filter genes based on mean, std and sample_frac. This also subsamples the resulting filtered gene list by self.rand_frac. 
#         # If self.gene_list_csv path is specified it works like a wildcard and CanDLE will train only with the genes in the csv path
#         self.filtered_gene_list, self.gene_filtered_data_matrix = self.filter_genes()
#         # Perform batch normalization, this uses self.general_stats and normalizes self.gene_filtered_data_matrix  
#         self.batch_normalize()
#         # # Filter self.label_df and self.lab_txt_2_lab_num based on the specified tissue # TODO: Add filter by tissue function
#         # self.filter_by_tissue()
#         # Make the problem binary in case self.binary_dict is not empty
#         self.make_binary_problem() # If self.binary_dict == {} this function does nothing
#         # Split data into train, validation and test sets. This function uses self.label_df to split the data with the same proportion.
#         self.split_labels, self.split_matrices = self.split_data() # For split_matrices samples are columns and genes are rows
#         # Define number of classes for classification
#         self.num_classes = len(self.lab_txt_2_lab_num.keys()) if self.binary_dict == {} else 2

#         # Plot relevant plots here # TODO: Incorporate all the plots from Toil here
#         # self.plot_dim_reduction()


#     def make_mappers(self):
#         """
#         This function generates mapper files useful for class definition in the dataset by running the make_mappers.py file
#         """
#         # Just make mappers if they are not already saved
#         if not os.path.exists(os.path.join(self.path, 'mappers', 'wang_standard_label_mapper.json')) or self.force_compute:
#             # run main.py with subprocess
#             command = f'python make_mappers.py'
#             print(command)
#             command = command.split()
#             subprocess.call(command)

#     # This function unzips the raw downloaded data from 
#     def unzip_data(self):
#         final_data_path = os.path.join(self.path, 'original_data')
#         # Do nothing if unzipped folder already exists
#         if os.path.exists(final_data_path) and (self.force_compute==False):
#             print('Files already unzipped...')
#             return
#         # Unzip data if original_data does not exist
#         else:
#             print('Unzipping files this may take some minutes...')
#             zipped_path = os.path.join(self.path, 'raw_data.zip')
#             unzipped_folder = os.path.join(self.path, 'raw_data_unzipped')
#             final_data_path = os.path.join(self.path, 'original_data')

#             with zipfile.ZipFile(zipped_path, 'r') as zip_ref:
#                 zip_ref.extractall(unzipped_folder)
            
#             classes_paths = os.listdir(unzipped_folder)
            
#             # Make final directory
#             os.makedirs(final_data_path, exist_ok=True)
#             # Cycle to unzip original data
#             for i in tqdm(range(len(classes_paths))):
#                 class_path = classes_paths[i]
#                 final_file_name = class_path[:-3]
#                 # Exclude chol category which is said to be discarded in original paper but is in the downloaded data
#                 if not(class_path[:4] == 'chol'):
#                 #if True:
#                     with gzip.open(os.path.join(unzipped_folder, class_path), 'rb') as f_in:
#                         with open(os.path.join(final_data_path, final_file_name), 'wb') as f_out:
#                             shutil.copyfileobj(f_in, f_out)
            
#             # Remove temporal folder
#             shutil.rmtree(unzipped_folder)
    
#     # This helper function receives the file name of a class and returns a valid textual label
#     def get_label_from_name(self, name):
#         str_list = name[:-4].split('-')
#         if len(str_list) == 5:
#             label = str_list[-2]+'-'+str_list[-1]+'-'+str_list[0]
#         elif len(str_list) == 4:
#             label = str_list[-1]+'-'+str_list[0]
#         else:
#             raise ValueError('The name of the original file is not adequate.')
#         label = label.upper()

#         return label

#     # This function reads the data
#     def read_data(self):
#         # If processed data directory does not exist read, merge and save complete data
#         if not os.path.exists(os.path.join(self.path, 'processed_data')) or (self.force_compute == True):
            
#             print(f'Reading data from {os.path.join(self.path, "original_data")}')
#             start = time.time()
#             data_path = os.path.join(self.path, 'original_data')
#             # Declare list of paths where each class is hosted
#             classes_paths = os.listdir(data_path)

#             for i in tqdm(range(len(classes_paths))):
#                 class_file = classes_paths[i] # Get file name
#                 act_df = pd.read_table(os.path.join(data_path, class_file), delimiter='\t') # Read file
                
#                 # Perform minor modification in act_df
#                 act_df = act_df.set_index('Hugo_Symbol') 
#                 del act_df['Entrez_Gene_Id']
                
#                 # Filter out genes in act_df that are not in all classes 
#                 valid_gene_index = act_df.index if i==0 else valid_gene_index.intersection(act_df.index)
#                 act_df = act_df.loc[valid_gene_index, :]

#                 act_category = self.get_label_from_name(class_file) # Get label names from file names
#                 act_category_df = pd.DataFrame({'sample':act_df.columns, 'original_lab': act_category}) # Put original label name
                
#                 # Join iteratively data matrices
#                 data_matrix = act_df if i==0 else data_matrix.join(act_df)
#                 data_matrix = data_matrix.loc[valid_gene_index, :] # Ensure data matrix just has common genes in all classes
#                 category_df = act_category_df if i==0 else pd.concat([category_df, act_category_df], axis=0) # Join category dataframes

#             # Sort Genes and samples in data matrix
#             data_matrix.sort_index(inplace=True) # Sort genes
#             data_matrix = data_matrix.T.sort_index().T # Sort samples
#             # Set samples as index and sort category dataframe
#             category_df.set_index('sample', inplace=True)
#             category_df.sort_index(inplace=True)


#             # Add a binary column to category_df indicating if the samples is from the TCGA
#             category_df['is_tcga'] = category_df['original_lab'].str.contains('TCGA')

#             # Loads standard label mapper
#             with open(os.path.join(self.path, "mappers", "wang_standard_label_mapper.json"), "r") as f:
#                 standard_label_mapper = json.load(f)

#             # Add a column with the standard labels shared between all datasets (Toil, Wang and Recount3)
#             category_df['lab_txt'] = category_df['original_lab'].map(standard_label_mapper)

#             # Print time that was needed to read the data
#             end = time.time()
#             print(f'Time to read data: {round(end-start,2)} s')

#             # Log2(x+1) transform the data
#             tqdm.pandas(desc="Computing Log2(x+1) transform")
#             data_matrix = data_matrix.progress_apply(lambda x: np.log2(x+1))
            
#             # Reset index to save in feather file
#             data_matrix.reset_index(inplace=True)

#             print(f'Saving processed data to {os.path.join(self.path, "processed_data")}')
#             os.makedirs(os.path.join(self.path, 'processed_data'), exist_ok=True)
#             data_matrix.to_feather(os.path.join(self.path, 'processed_data', 'data_matrix.feather'))
#             category_df.to_csv(os.path.join(self.path, 'processed_data', 'data_category.csv'))

#             # Set index again for next steps
#             data_matrix.set_index('Hugo_Symbol', inplace=True)

            
#         # If the data is already merged and stored load it from file
#         else:
#             print(f'Loading processed data from {os.path.join(self.path, "processed_data")}')
#             start = time.time()
#             data_matrix = pd.read_feather(os.path.join(self.path, 'processed_data', 'data_matrix.feather'))
#             category_df = pd.read_csv(os.path.join(self.path, 'processed_data', 'data_category.csv'), index_col='sample')
#             end = time.time()
#             print(f'Time to read data: {round(end-start,2)} s')
#             data_matrix.set_index('Hugo_Symbol', inplace=True)

#         return data_matrix, category_df

#     # Filters the Wang data set by using or not using TCGA and GTEx samples.
#     def filter_datasets(self):

#         tcga_samples = self.categories.index[self.categories['is_tcga']]
#         gtex_samples = self.categories.index[~self.categories['is_tcga']]

#         # Handle the filters for TCGA and GTEx
#         if self.dataset == 'tcga':
#             print("Using TCGA samples only")
#             # Filter out all gtex samples from matrix_data
#             matrix_data_filtered = self.matrix_data.loc[:, tcga_samples]
#             categories_filtered = self.categories.loc[tcga_samples, :]
#         elif self.dataset == 'gtex':
#             print("Using GTEx samples only")
#             # Filter out all tcga samples from matrix_data
#             matrix_data_filtered = self.matrix_data.loc[:, gtex_samples]
#             categories_filtered = self.categories.loc[gtex_samples, :]
#         elif self.dataset == 'both':
#             # Do nothing because both TCGA and GTEX samples are included
#             print("Using TCGA and GTEX samples")
#             matrix_data_filtered = self.matrix_data
#             categories_filtered = self.categories
            
#         return matrix_data_filtered, categories_filtered

#     # This function extracts the labels from categories and returns a label dataframe and a dictionary of textual labels to numeric labels
#     def find_labels(self):

#         # Make dataset directory if it does not exist
#         os.makedirs(self.dataset_info_path, exist_ok = True)

#         # Initialize label_df as filtered categories 
#         label_df = self.categories_filtered

#         # Handle the only TCGA case where all normal samples have to be grouped in a single NT class
#         if self.dataset == 'tcga':
#             label_df.loc[label_df['lab_txt'].str.contains('GTEX'), 'lab_txt'] = 'TCGA-NT'

#         # Get unique textual labels obtained and sort them
#         current_labels = sorted(label_df["lab_txt"].unique().tolist())
#         # Define lab_txt_2_lab_num dictionary
#         lab_txt_2_lab_num = {lab_txt: i for i, lab_txt in enumerate(current_labels)}

#         # Define numeric labels from the textual labels in label_df
#         label_df["lab_num"] = label_df["lab_txt"].map(lab_txt_2_lab_num)
        
#         # Save lab_txt_2_lab_num dictionary to json file
#         with open(os.path.join(self.dataset_info_path, "lab_txt_2_lab_num_mapper.json"), "w") as f:
#             json.dump(lab_txt_2_lab_num, f, indent = 4)

#         return label_df, lab_txt_2_lab_num

#     # TODO: Make that general stats is not hosted in self.path but in self.path/processed_data
#     # This function finds the mean expression, std and expressed sample fraction for GTEx, TCGA, healthy TCGA and the joint dataset
#     def find_general_stats(self):
#         # If the info stats are already computed load them from file
#         if (os.path.exists(os.path.join(self.path, 'general_stats.csv'))) & (self.force_compute == False):
#             print('Loading general stats from '+os.path.join(self.path, 'general_stats.csv'))
#             general_stats = pd.read_csv(os.path.join(self.path, 'general_stats.csv'), index_col = 0)
#         # If the stats are not computed compute them and save them in file
#         else:
#             print('Computing general stats and saving to '+os.path.join(self.path, 'general_stats.csv'))
#             # Define auxiliary tcga dataframe to obtain healthy tcga samples
#             tcga_df = self.label_df[self.label_df['is_tcga']]

#             # Get the identifiers of the samples in each subset
#             gtex_samples = self.label_df[self.label_df['is_tcga']==False].index
#             tcga_samples = tcga_df.index
#             healthy_tcga_samples = tcga_df[tcga_df['lab_txt'].str.contains('GTEX')].index
#             joint_samples = self.label_df.index

#             # Compute the mean of the subsets
#             tqdm.pandas(desc="Computing Mean GTEx")
#             gtex_mean = self.matrix_data.loc[:, gtex_samples].progress_apply(np.mean, axis = 1).to_frame(name='gtex_mean')
#             tqdm.pandas(desc="Computing Mean TCGA")
#             tcga_mean = self.matrix_data.loc[:, tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='tcga_mean')
#             tqdm.pandas(desc="Computing Mean Healthy TCGA")
#             healthy_tcga_mean = self.matrix_data.loc[:, healthy_tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='healthy_tcga_mean')
#             tqdm.pandas(desc="Computing Joint Mean")
#             joint_mean = self.matrix_data.loc[:, joint_samples].progress_apply(np.mean, axis = 1).to_frame(name='joint_mean')

#             # Compute the std of the subsets
#             tqdm.pandas(desc="Computing std GTEx")
#             gtex_std = self.matrix_data.loc[:, gtex_samples].progress_apply(np.std, axis = 1).to_frame(name='gtex_std')
#             tqdm.pandas(desc="Computing std TCGA")
#             tcga_std = self.matrix_data.loc[:, tcga_samples].progress_apply(np.std, axis = 1).to_frame(name='tcga_std')
#             tqdm.pandas(desc="Computing std Healthy TCGA")
#             healthy_tcga_std = self.matrix_data.loc[:, healthy_tcga_samples].progress_apply(np.std, axis = 1).to_frame(name='healthy_tcga_std')
#             tqdm.pandas(desc="Computing Joint std")
#             joint_std = self.matrix_data.loc[:, joint_samples].progress_apply(np.std, axis = 1).to_frame(name='joint_std')

#             # Compute the fraction of samples where a gene is expressed
#             print('Computing fraction of samples where each gene is expressed ...')
#             min_val = self.matrix_data.min().min() # Get minimum value
#             tqdm.pandas(desc="Computing Expressed Genes")
#             expressed_matrix = self.matrix_data.progress_apply(lambda x: x>min_val, axis = 1) # Compute expressed positions
            
#             # Compute expressed sample fractions for all subsets
#             tqdm.pandas(desc="Computing Joint Expressed Sample Fraction")
#             joint_sample_fraction = expressed_matrix.progress_apply(np.mean, axis = 1).to_frame(name='joint_sample_frac')
#             tqdm.pandas(desc="Computing GTEx Expressed Sample Fraction")
#             gtex_sample_fraction = expressed_matrix.loc[:, gtex_samples].progress_apply(np.mean, axis = 1).to_frame(name='gtex_sample_frac')
#             tqdm.pandas(desc="Computing TCGA Expressed Sample Fraction")
#             tcga_sample_fraction = expressed_matrix.loc[:, tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='tcga_sample_frac')
            

#             # Join stats in single dataframe
#             general_stats = pd.concat([gtex_mean, tcga_mean, healthy_tcga_mean, joint_mean,
#                                         gtex_std, tcga_std, healthy_tcga_std, joint_std,
#                                         joint_sample_fraction, gtex_sample_fraction, tcga_sample_fraction,], axis=1)
#             general_stats.to_csv(os.path.join(self.path, 'general_stats.csv'))

#         return general_stats

#     # This function filters out genes by mean, standard deviation, expression fraction, random fraction or list of genes
#     def filter_genes(self):
#         # If there is a gene list specified by parameter then it overwrites mean, std and rand_frac filtering  
#         if self.gene_list_csv != 'None':
#             # Print user message
#             print(f'CanDLE will train with the list of genes specified in {self.gene_list_csv}')
#             gene_csv_df = pd.read_csv(self.gene_list_csv, index_col=0)
#             gene_list = pd.Index(gene_csv_df['gene_name'])
        
#         # If no list of genes is specified then proceed with mean, std, sample_frac and rand_frac filtering
#         elif (not os.path.exists(os.path.join(self.dataset_info_path, "filtering_info.csv"))) or self.force_compute:
            
#             print("Computing list of filtered genes. And saving filtering info to:\n\t"+ os.path.join(self.dataset_info_path, "filtering_info.csv"))
            
#             # Find the indices of the samples with mean, standard deviation and sample fractions that fulfill the thresholds
#             mean_bool_index = ((self.general_stats['joint_mean']>self.mean_thr) & (self.general_stats['gtex_mean']>self.mean_thr) & (self.general_stats['tcga_mean']>self.mean_thr))
#             std_bool_index = ((self.general_stats['joint_std']>self.std_thr) & (self.general_stats['gtex_std']>self.std_thr) & (self.general_stats['tcga_std']>self.std_thr))
#             sample_frac_bool_index = ((self.general_stats['joint_sample_frac'] > self.sample_frac) & (self.general_stats['gtex_sample_frac'] > self.sample_frac) & (self.general_stats['tcga_sample_frac'] > self.sample_frac))
            
#             # Compute intersection of mean_data_index and std_data_index
#             mean_std_sample_index = np.logical_and.reduce((mean_bool_index.values, std_bool_index.values, sample_frac_bool_index)).ravel()
#             # Make a gene list of the samples that fulfill the thresholds
#             gene_list = self.matrix_data.index[mean_std_sample_index]

#             # Subsample gene list in case self.rand_frac < 1
#             if self.rand_frac < 1:
#                 np.random.seed(0) # Ensure reproducibility # TODO: Parametrize this seed to run variation experiments
#                 rand_selector = np.zeros(len(gene_list))
#                 rand_selector[:int(len(gene_list)*self.rand_frac)] = 1
#                 np.random.shuffle(rand_selector) # Shuffle boolean selector
#                 rand_selector = np.array(rand_selector, dtype=bool)
#                 gene_list = gene_list[rand_selector] # Filter gene list based in rand_selector
            
#             # Compute boolean value for each gene that indicates if it was included in the filtered gene list
#             included_in_filtering = self.general_stats.index.isin(gene_list)

#             # Merge all statistics and included_in_filtering into a final dataframe
#             filtering_info_df = self.general_stats
#             filtering_info_df['included'] = included_in_filtering
#             filtering_info_df.index.name = "gene"

#             # Save the filtering info to files
#             filtering_info_df.to_csv(os.path.join(self.dataset_info_path, "filtering_info.csv"), index = True)
#             # Plot histograms with plot_filtering_histograms()
#             self.plot_filtering_histograms(filtering_info_df)

#         else:
#             print("Loading filtering info from:\n\t" + os.path.join(self.dataset_info_path, "filtering_info.csv"))
#             filtering_info_df = pd.read_csv(os.path.join(self.dataset_info_path, "filtering_info.csv"), index_col = 0)
#             # get indices of filtering_info_df that are True in the included column
#             gene_list = filtering_info_df.index[filtering_info_df["included"].values == True]
#             # Plot histograms with plot_filtering_histograms()
#             self.plot_filtering_histograms(filtering_info_df)
        
#         # Filter tha data matrix based on the gene list
#         gene_filtered_data_matrix = self.matrix_data_filtered.loc[gene_list, :]

#         print("Currently working with {} genes...".format(gene_filtered_data_matrix.shape[0]))

#         return gene_list.to_list(), gene_filtered_data_matrix
    
#     # This function performs a data normalization by batches (GTEX or TCGA) 
#     def batch_normalize(self):
#         if self.batch_normalization=='None':
#             print('Did not perform batch normalization...')
#             return
#         else:
#             print('Batch normalizing matrix data...')
#             start = time.time()
#             # Get the identifiers of the samples in each subset
#             gtex_samples = self.label_df[self.label_df['is_tcga']==False].index
#             tcga_samples = self.label_df[self.label_df['is_tcga']==True].index
#             # Get stats of the valid genes
#             valid_stats = self.general_stats.loc[self.filtered_gene_list, :]

#             # Transforms GTEx data
#             normalized_gtex = self.gene_filtered_data_matrix[gtex_samples].sub(valid_stats['gtex_mean'], axis=0)
#             normalized_gtex = normalized_gtex.div(valid_stats['gtex_std'], axis=0)
           
#            # Transform TCGA data according to self.batch_normalization
#             if self.batch_normalization=='normal':
#                 normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['tcga_mean'], axis=0)
#                 normalized_tcga = normalized_tcga.div(valid_stats['tcga_std'], axis=0)
            
#             elif self.batch_normalization=='healthy_tcga':
#                 normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['healthy_tcga_mean'], axis=0)
#                 normalized_tcga = normalized_tcga.div(valid_stats['healthy_tcga_std'], axis=0)
            
#             else:
#                 raise ValueError('Batch normalization should be None, normal or healthy_tcga.')

#             normalized_joint = pd.concat([normalized_gtex, normalized_tcga], axis=1)

#             # Sort columns of normalized joint
#             normalized_joint = normalized_joint.T.sort_index().T

#             # Replace NaNs generated by std division to 0's
#             self.gene_filtered_data_matrix = normalized_joint.fillna(0.0)
#             end = time.time()
#             print(f'It took {round(end-start, 2)} s to batch normalize the data.')

#     # This function uses self.binary_dict to modify self.label_df and self.lab_txt_2_lab_num to make the labels binary
#     def make_binary_problem(self):
#         # If binary_dict is not specified, do not make binary
#         if self.binary_dict == {}:
#             print("No binary problem specified.")
#             return
#         # If binary_dict is specified, make binary
#         else:
#             self.lab_txt_2_lab_num = self.binary_dict
#             # Define numeric labels from the textual labels in self.label_df
#             self.label_df["lab_num"] = self.label_df["lab_txt"].map(self.lab_txt_2_lab_num)   
#             print("Made binary problem.")
#             print(f"Number of samples in class 0: {len(self.label_df[self.label_df['lab_num'] == 0])}, number of samples in class 1: {len(self.label_df[self.label_df['lab_num'] == 1])}")
#             return

#     # This function uses self.label_df to split the data into train, validation and test sets
#     def split_data(self):
#         train_val_lab, test_lab = train_test_split(self.label_df["lab_num"], test_size = 0.2, random_state = self.partition_seed, stratify = self.label_df["lab_num"].values)
#         train_lab, val_lab = train_test_split(train_val_lab, test_size = 0.25, random_state = self.partition_seed, stratify = train_val_lab.values)
#         # Use label indexes to subset the data in self.matrix_data_filtered
#         train_matrix = self.gene_filtered_data_matrix[train_lab.index]
#         val_matrix = self.gene_filtered_data_matrix[val_lab.index]
#         test_matrix = self.gene_filtered_data_matrix[test_lab.index]
#         train_val_matrix = self.gene_filtered_data_matrix[train_val_lab.index]
#         # Declare label dictionaries
#         split_labels = {"train": train_lab, "val": val_lab, "test": test_lab, "train_val": train_val_lab}
#         # Declare matrix dictionaries
#         split_matrices = {"train": train_matrix, "val": val_matrix, "test": test_matrix, "train_val": train_val_matrix}
#         # Both matrixes and labels are already shuffled
#         return split_labels, split_matrices

#     # This function gets the dataloaders for the train, val and test sets
#     def get_dataloaders(self, batch_size):
#         # Select data partitions
#         # These data matrices have samples in rows and genes in columns
#         x_train = torch.Tensor(self.split_matrices["train"].T.values).type(torch.float)
#         x_val = torch.Tensor(self.split_matrices["val"].T.values).type(torch.float)
#         x_test = torch.Tensor(self.split_matrices["test"].T.values).type(torch.float)
        
#         # Cast labels as tensors
#         y_train = torch.Tensor(self.split_labels["train"].values).type(torch.long)
#         y_val = torch.Tensor(self.split_labels["val"].values).type(torch.long)
#         y_test = torch.Tensor(self.split_labels["test"].values).type(torch.long)

#         # Define train, val and test datasets
#         train_dataset = TensorDataset(x_train, y_train)
#         val_dataset = TensorDataset(x_val, y_val)
#         test_dataset = TensorDataset(x_test, y_test)

#         # Create dataloaders
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#         return train_loader, val_loader, test_loader

#     # This function plots a 2X2 figure with the histograms of mean expression and standard deviation before and after filtering
#     def plot_filtering_histograms(self, filtering_info_df):        
#         # Make a figure
#         fig, axes = plt.subplots(2, 2, figsize = (18, 12))
#         fig.suptitle("Filtering of Mean > " +str(self.mean_thr) + " and Standard Deviation > " + str(self.std_thr) , fontsize = 30)
#         # Variable to adjust display height of the histograms
#         max_hist = np.zeros((2, 2))

#         # Plot the histograms of mean and standard deviation before filtering
#         n, _, _ = axes[0, 0].hist(filtering_info_df["joint_mean"], bins = 50, color = "k", density=True)
#         max_hist[0, 0] = np.max(n)
#         axes[0, 0].set_title("Before filtering", fontsize = 26)        
#         n, _, _ = axes[1, 0].hist(filtering_info_df["joint_std"], bins = 50, color = "k", density=True)
#         max_hist[1, 0] = np.max(n)
#         # Plot the histograms of mean and standard deviation after filtering
#         n, _, _ = axes[0, 1].hist(filtering_info_df["joint_mean"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
#         max_hist[0, 1] = np.max(n)
#         axes[0, 1].set_title("After filtering", fontsize = 26)
#         n, _, _ = axes[1, 1].hist(filtering_info_df["joint_std"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
#         max_hist[1, 1] = np.max(n)

#         # Format axes
#         for i in range(2):
#             for j in range(2):
#                 axes[i, j].set_ylabel("Density", fontsize = 16)
#                 axes[i, j].tick_params(labelsize = 14)
#                 axes[i, j].grid(True)
#                 axes[i, j].set_axisbelow(True)
#                 axes[i, j].set_ylim(0, max_hist[i, j] * 1.1)
#                 # Handle mean expression plots
#                 if i == 0:
#                     axes[i, j].set_xlabel("Mean expression", fontsize = 16)
#                     axes[i, j].set_xlim(filtering_info_df["joint_mean"].min(), filtering_info_df["joint_mean"].max())
#                     axes[i, j].plot([self.mean_thr, self.mean_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")
#                 # Handle standard deviation plots
#                 else:
#                     axes[i, j].set_xlabel("Standard deviation", fontsize = 16)
#                     axes[i, j].set_xlim(filtering_info_df["joint_std"].min(), filtering_info_df["joint_std"].max())
#                     axes[i, j].plot([self.std_thr, self.std_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")

#         # Save the figure
#         fig.savefig(os.path.join(self.dataset_info_path, "filtering_histograms.png"), dpi = 300)
#         plt.close(fig)

#     def plot_dim_reduction(self):
        
#         valid_samples = self.gene_filtered_data_matrix.columns
#         valid_genes = self.filtered_gene_list
#         bool_sample_index = self.matrix_data.columns.isin(valid_samples)
#         bool_gene_index = self.matrix_data.index.isin(valid_genes)
#         print('Filtering raw data to valid genes and samples...')
#         raw_data = self.matrix_data.loc[bool_gene_index, bool_sample_index].T.sort_index()
#         processed_data = self.gene_filtered_data_matrix.T.sort_index()
#         # Get and sort metadata
#         meta_df = self.label_df
#         meta_df = meta_df.sort_index()

#         if (not os.path.exists(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'))) or self.force_compute:
            
#             print(f'Computing dimensionality reduction and saving it to {os.path.join(self.dataset_info_path, "dimensionality_reduction.pkl")}...')
            
#             print('Reducing dimensions with PCA...')
#             raw_pca = PCA(n_components=2, random_state=0)
#             processed_pca = PCA(n_components=2, random_state=0)
#             r_raw_pca = raw_pca.fit_transform(raw_data)
#             r_processed_pca = processed_pca.fit_transform(processed_data)

#             # If this shows a warning with OpenBlast you can solve it using this GitHub issue https://github.com/ultralytics/yolov5/issues/2863
#             # You just need to write in terminal "export OMP_NUM_THREADS=1"
#             print('Reducing dimensions with TSNE...')
#             raw_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='random', n_jobs=-1, verbose=2)       # learning_rate and init were set due to a future warning
#             processed_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='random', n_jobs=-1, verbose=2)
#             r_raw_tsne = raw_tsne.fit_transform(raw_data)
#             r_processed_tsne = processed_tsne.fit_transform(processed_data)

#             print('Reducing dimensions with UMAP...')
#             raw_umap = UMAP(n_components=2, random_state=0) # , n_neighbors=64, local_connectivity=32)          # n_neighbors and local_connectivity are set to ensure that the graph is connected
#             processed_umap = UMAP(n_components=2, random_state=0) # , n_neighbors=64, local_connectivity=32)
#             r_raw_umap = raw_umap.fit_transform(raw_data)
#             r_processed_umap = processed_umap.fit_transform(processed_data)

#             reduced_dict = {'raw_pca': r_raw_pca,               'raw_tsne': r_raw_tsne,                 'raw_umap': r_raw_umap,
#                             'processed_pca': r_processed_pca,   'processed_tsne': r_processed_tsne,     'processed_umap':  r_processed_umap}

#             with open(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'), 'wb') as f:
#                 pkl.dump(reduced_dict, f)
        
#         else:
            
#             print(f'Loading dimensionality reduction from {os.path.join(self.dataset_info_path, "dimensionality_reduction.pkl")}...')
            
#             with open(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'), 'rb') as f:
#                 reduced_dict = pkl.load(f)
        

#         def plot_dim_reduction(reduced_dict, meta_df, color_type='tcga', cmap=None):


#             # Load id_2_tissue mapper from file
#             with open(os.path.join(self.path, "mappers", "id_2_tissue_mapper.json"), "r") as f:
#                 id_2_tissue_mapper = json.load(f)

#             meta_df['tissue'] = meta_df['lab_txt'].map(id_2_tissue_mapper)

#             # Get dictionaries to have different options to colorize the scatter points
#             tcga_dict = {True: 1, False: 0}
#             tissue_dict = {tissue: i/len(meta_df.tissue.unique()) for i, tissue in enumerate(sorted(meta_df.tissue.unique()))}
#             class_dict = {cl: i/len(meta_df.lab_txt.unique()) for i, cl in enumerate(sorted(meta_df.lab_txt.unique()))}

#             # Define color map
#             if cmap is None:
#                 d_colors = ["black", "darkcyan"]
#                 cmap = LinearSegmentedColormap.from_list("candle_cmap", d_colors)
#             else:
#                 cmap = get_cmap(cmap)
            
#             # Compute color values
#             if color_type == 'tcga':
#                 meta_df['color'] = meta_df['is_tcga'].map(tcga_dict)
#             elif color_type == 'tissue':
#                 meta_df['color'] = meta_df['tissue'].map(tissue_dict)
#             elif color_type == 'class':
#                 meta_df['color'] = meta_df['lab_txt'].map(class_dict)

#             # Plot figure
#             fig, ax = plt.subplots(2, 3)

#             ax[0, 0].scatter(reduced_dict['raw_pca'][:,0],          reduced_dict['raw_pca'][:,1],           c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.1)
#             ax[1, 0].scatter(reduced_dict['processed_pca'][:,0],    reduced_dict['processed_pca'][:,1],     c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.1)
#             ax[0, 1].scatter(reduced_dict['raw_tsne'][:,0],         reduced_dict['raw_tsne'][:,1],          c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.1)
#             ax[1, 1].scatter(reduced_dict['processed_tsne'][:,0],   reduced_dict['processed_tsne'][:,1],    c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.1)
#             ax[0, 2].scatter(reduced_dict['raw_umap'][:,0],         reduced_dict['raw_umap'][:,1],          c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.1)
#             ax[1, 2].scatter(reduced_dict['processed_umap'][:,0],   reduced_dict['processed_umap'][:,1],    c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.1)

#             label_dict = {0: 'PC', 1: 'T-SNE', 2: 'UMAP'}
#             title_dict = {0: 'PCA', 1: 'T-SNE', 2: 'UMAP'}

#             for i in range(ax.shape[0]):
#                 for j in range(ax.shape[1]):
#                     ax[i,j].spines.right.set_visible(False) 
#                     ax[i,j].spines.top.set_visible(False)
#                     ax[i,j].set_xlabel(f'{label_dict[j]}1')
#                     ax[i,j].set_ylabel(f'{label_dict[j]}2')
#                     ax[i,j].set_title(f'{title_dict[j]} of Raw Data' if i==0 else f'{title_dict[j]} of Processed Data')

#             # Set the legend
#             if (color_type == 'tcga'):
#                 handles = [mpatches.Patch(facecolor=cmap(0.0), edgecolor='black', label='GTEx'),
#                            mpatches.Patch(facecolor=cmap(1.0), edgecolor='black', label='TCGA')]
#                 fig.set_size_inches(13, 7)
#                 ax[0, 2].legend(handles=handles, loc='center left',bbox_to_anchor=(1.15, 0.49))
#                 plt.tight_layout()

#             elif (color_type == 'tissue'):
#                 handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'{key} Tissue') for key, val in tissue_dict.items()]
#                 fig.set_size_inches(15, 7)
#                 fig.legend(handles=handles, loc=7)
#                 fig.tight_layout()
#                 fig.subplots_adjust(right=0.83)
                

#             elif (color_type == 'class'):
#                 handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'{key}') for key, val in class_dict.items()]
#                 fig.set_size_inches(18, 8)
#                 fig.legend(handles=handles, loc=7, ncol=2)
#                 fig.tight_layout()
#                 fig.subplots_adjust(right=0.75)

            
#             else:
#                 raise ValueError('Invalid color type...')

#             # Save
#             fig.savefig(os.path.join(self.dataset_info_path, f'dim_reduction_{color_type}.png'), dpi=300)
#             plt.close()

#         # Plot dimensionality reduction with the three different color styles
#         plot_dim_reduction(reduced_dict, meta_df, color_type='tcga')
#         plot_dim_reduction(reduced_dict, meta_df, color_type='tissue', cmap='brg')
#         plot_dim_reduction(reduced_dict, meta_df, color_type='class', cmap='brg')





# class Recount3Dataset():
#     def __init__(self, path, dataset = 'both', tissue='all', binary_dict={}, mean_thr=-10.0,
#                 std_thr=0.01, rand_frac = 1.0, sample_frac = 0.5, gene_list_csv='None',
#                 batch_normalization='None', partition_seed=0, force_compute = False):

#         self.path = path
#         self.tissue = tissue
#         self.binary_dict = binary_dict
#         self.dataset = dataset
#         self.dataset_info_path = os.path.join(self.path, 'processed_data',
#                                               f'dataset={dataset}',
#                                               f'mean_thr={mean_thr}_std_thr={std_thr}',
#                                               f'sample_frac={sample_frac}_rand_frac={rand_frac}', 
#                                               f'tissue={self.tissue}')
#         self.mean_thr = mean_thr
#         self.std_thr = std_thr
#         self.rand_frac = rand_frac
#         self.sample_frac = sample_frac
#         self.gene_list_csv = gene_list_csv
#         self.batch_normalization = batch_normalization
#         self.partition_seed = partition_seed # seed for train/val/test split
#         self.force_compute = force_compute

#         # Main Bioinformatic pipeline
#         # Make mapper files if they are not already saved
#         self.make_mappers()
#         # Read data from the Recount3 dataset and perform a log2(x+1) transformation. Also return gene metadata that is only available for Recount3
#         self.matrix_data, self.categories, self.gene_meta = self.read_data()
#         # Filter Wang datasets to use GTEx, TCGA or both.
#         self.matrix_data_filtered, self.categories_filtered = self.filter_datasets()
#         # Get labels dataframe and label dictionary. 
#         self.label_df, self.lab_txt_2_lab_num = self.find_labels()
#         # Find stats of each dataset segment
#         self.general_stats = self.find_general_stats()
#         # Filter genes based on mean, std and sample_frac. This also subsamples the resulting filtered gene list by self.rand_frac. 
#         # If self.gene_list_csv path is specified it works like a wildcard and CanDLE will train only with the genes in the csv path
#         self.filtered_gene_list, self.gene_filtered_data_matrix = self.filter_genes()
#         # Perform batch normalization, this uses self.general_stats and normalizes self.gene_filtered_data_matrix  
#         self.batch_normalize()
#         # # Filter self.label_df and self.lab_txt_2_lab_num based on the specified tissue # TODO: Add filter by tissue function
#         # self.filter_by_tissue()
#         # Make the problem binary in case self.binary_dict is not empty
#         self.make_binary_problem() # If self.binary_dict == {} this function does nothing
#         # Split data into train, validation and test sets. This function uses self.label_df to split the data with the same proportion.
#         self.split_labels, self.split_matrices = self.split_data() # For split_matrices samples are columns and genes are rows

#         # Define number of classes for classification
#         self.num_classes = len(self.lab_txt_2_lab_num.keys()) if self.binary_dict == {} else 2

#         # Plot relevant plots here # TODO: Incorporate all the plots from Toil here
#         # self.plot_dim_reduction()

#     def make_mappers(self):
#         """
#         This function generates mapper files useful for class definition in the dataset by running the make_mappers.py file
#         """
#         # Just make mappers if they are not already saved
#         if not os.path.exists(os.path.join(self.path, 'mappers', 'recount3_standard_label_mapper.json')) or self.force_compute:
#             # run main.py with subprocess
#             command = f'python make_mappers.py'
#             print(command)
#             command = command.split()
#             subprocess.call(command)

#     def read_data(self):
#         """
#         Reads data from the Recount3 data set with root path and returns matrix_data and categories dataframes.

#         Returns:
#             matrix_data (pd.dataframe): Matrix data of the Recount3 data set. Columns are samples and rows are genes.
#             categories (pd.dataframe): Categories of the Recount3 data set. Rows are samples. Columns are 'lab_txt', 'is_tcga' and 'healthy'
#             gene_meta (pd.DataFrame): Useful information about the available genes.
#         """
#         start = time.time()

#         if (not os.path.exists(os.path.join(self.path, "processed_data"))) or self.force_compute:
#             print(f'Reading data from {os.path.join(self.path, "original_data")} and performing processing and transformations...')
#             # Read all data
#             matrix_data = pd.read_feather(os.path.join(self.path, 'original_data', "data_matrix.feather"))
#             gtex_meta = pd.read_csv(os.path.join(self.path, 'original_data', "gtex_metadata.csv"), low_memory=False)
#             tcga_meta = pd.read_csv(os.path.join(self.path, 'original_data', "tcga_metadata.csv"), low_memory=False)
#             gene_meta = pd.read_csv(os.path.join(self.path, 'original_data', "gene_metadata.csv"), low_memory=False)
            
#             # Process TCGA metadata #############
#             # Filter just important TCGA metadata
#             # TODO: Sample identifiers here are weird fot TCGA metadata and data matrix. It works but it would be better to standardize identifiers with wang and toil
#             tcga_meta = tcga_meta[['Unnamed: 0', 'tcga.gdc_cases.project.project_id', 'tcga.gdc_cases.project.primary_site', 'tcga.cgc_sample_sample_type']]
#             # Rename columns
#             tcga_meta.columns = ['sample', 'lab_tcga', 'tissue', 'sample_type']
#             # Get a column of the tcga_meta indicating if the data is healthy
#             tcga_meta['healthy'] = tcga_meta['sample_type'] == 'Solid Tissue Normal'
#             # Reset index
#             tcga_meta.set_index('sample', inplace=True)
#             # Filter out rows with NaNs
#             tcga_meta.dropna(inplace=True)
#             # Add is_tcga column
#             tcga_meta['is_tcga'] = True

#             # Get standard lab_txt labels from tcga_meta #######
#             # Initialize the lab_txt column as the TCGA projects
#             tcga_meta['lab_txt'] = tcga_meta['lab_tcga']
#             # Get mapper dict for healthy TCGA samples
#             with open(os.path.join(self.path, "mappers", "healthy_tcga_2_gtex_mapper.json"), 'r') as f:
#                 normal_tcga_mapper = json.load(f)
#             # Modify lab_txt labels of healthy TCGA samples to corresponded GTEx labels
#             tcga_meta.loc[tcga_meta['healthy'], 'lab_txt'] = tcga_meta[tcga_meta['healthy']]['tissue'].map(normal_tcga_mapper)
            

#             # Process GTEx metadata #############
#             # TODO: Standardize identifiers with wang and toil
#             gtex_meta = gtex_meta[['Unnamed: 0','gtex.smts','gtex.smtsd']]
#             # Rename columns
#             gtex_meta.columns = ['sample', 'lab_gtex', 'tissue']
#             # Obtain healthy column
#             gtex_meta['healthy'] = gtex_meta['tissue'] != 'Cells - Leukemia cell line (CML)' # FIXME: Finally decide if this cell lines are healthy or not. For now they are considered desease
#             # Reset index
#             gtex_meta.set_index('sample', inplace=True)
#             # Filter out rows with NaNs
#             gtex_meta.dropna(inplace=True)
#             # Add is_tcga column
#             gtex_meta['is_tcga'] = False
            
#             # Get standard lab_txt labels from gtex_meta #######
#             with open(os.path.join(self.path, "mappers", "recount3_gtex_mapper.json"), 'r') as f:
#                 gtex_mapper = json.load(f)
#             gtex_meta['lab_txt'] = gtex_meta['tissue'].map(gtex_mapper)

#             # Merge both metadata in single dataframe ##########
#             global_meta = pd.concat((gtex_meta[['lab_txt', 'is_tcga', 'healthy']], tcga_meta[['lab_txt', 'is_tcga', 'healthy']]))

#             # Filter matrix data to leave only the samples with valid metadata. This line also ensures that the ordering
#             # of metadata samples and matrix_data samples is the same
#             matrix_data = matrix_data[global_meta.index]

#             # Put gene ids in matrix_data index
#             matrix_data.set_index(gene_meta['gene_id'], inplace=True)

#             # Perform Log2(x+1) transformation over matrix_data
#             tqdm.pandas(desc="Computing Log2(x+1) transform")
#             matrix_data = matrix_data.progress_apply(lambda x: np.log2(x+1))

#             # Reset index to save in feather file
#             matrix_data.reset_index(inplace=True)

#             print(f'Saving processed data to {os.path.join(self.path, "processed_data")}')
#             os.makedirs(os.path.join(self.path, 'processed_data'), exist_ok=True)
#             matrix_data.to_feather(os.path.join(self.path, 'processed_data', 'data_matrix.feather'))
#             global_meta.to_csv(os.path.join(self.path, 'processed_data', 'data_category.csv'))

#             # Set index again for next steps
#             matrix_data.set_index('gene_id', inplace=True)
        
#         else:
#             print(f'Loading processed data from {os.path.join(self.path, "processed_data")}')
#             matrix_data = pd.read_feather(os.path.join(self.path, 'processed_data', 'data_matrix.feather'))
#             global_meta = pd.read_csv(os.path.join(self.path, 'processed_data', 'data_category.csv'), index_col='sample')
#             gene_meta = pd.read_csv(os.path.join(self.path, 'original_data', "gene_metadata.csv"), low_memory=False)
#             matrix_data.set_index('gene_id', inplace=True)

#         # Print the time needed to read and process raw data 
#         end = time.time()
#         print("Time to load data: {} s".format(round(end - start, 3)))
#         return matrix_data, global_meta, gene_meta

#     # Filters the Recount3 data set by using or not using TCGA and GTEx samples.
#     def filter_datasets(self):

#         tcga_samples = self.categories.index[self.categories['is_tcga']]
#         gtex_samples = self.categories.index[~self.categories['is_tcga']]

#         # Handle the filters for TCGA and GTEx
#         if self.dataset == 'tcga':
#             print("Using TCGA samples only")
#             # Filter out all gtex samples from matrix_data
#             matrix_data_filtered = self.matrix_data.loc[:, tcga_samples]
#             categories_filtered = self.categories.loc[tcga_samples, :]
#         elif self.dataset == 'gtex':
#             print("Using GTEx samples only")
#             # Filter out all tcga samples from matrix_data
#             matrix_data_filtered = self.matrix_data.loc[:, gtex_samples]
#             categories_filtered = self.categories.loc[gtex_samples, :]
#         elif self.dataset == 'both':
#             # Do nothing because both TCGA and GTEX samples are included
#             print("Using TCGA and GTEX samples")
#             matrix_data_filtered = self.matrix_data
#             categories_filtered = self.categories
            
#         return matrix_data_filtered, categories_filtered

#     # This function extracts the labels from categories and returns a label dataframe and a dictionary of textual labels to numeric labels
#     def find_labels(self):

#         # FIXME: This erases the dimensionality reduction files from previous experiments
#         # Make dataset directory if it does not exist
#         os.makedirs(self.dataset_info_path, exist_ok = True)

#         # Initialize label df with filtered categories
#         label_df = self.categories_filtered

#         # Handle the only TCGA case where all normal samples have to be grouped in a single NT class
#         if self.dataset == 'tcga':
#             label_df.loc[label_df['lab_txt'].str.contains('GTEX'), 'lab_txt'] = 'TCGA-NT'

#         # Get unique textual labels obtained and sort them
#         current_labels = sorted(label_df["lab_txt"].unique().tolist())
#         # Define lab_txt_2_lab_num dictionary
#         lab_txt_2_lab_num = {lab_txt: i for i, lab_txt in enumerate(current_labels)}

#         # Define numeric labels from the textual labels in label_df
#         label_df["lab_num"] = label_df["lab_txt"].map(lab_txt_2_lab_num)
        
#         # Save lab_txt_2_lab_num dictionary to json file
#         with open(os.path.join(self.dataset_info_path, "lab_txt_2_lab_num_mapper.json"), "w") as f:
#             json.dump(lab_txt_2_lab_num, f, indent = 4)

#         return label_df, lab_txt_2_lab_num


#     # TODO: Make that general stats is not hosted in self.path but in self.path/processed_data
#     # This function finds the mean expression, std and expressed sample fraction for GTEx, TCGA, healthy TCGA and the joint dataset
#     def find_general_stats(self):
#         # If the info stats are already computed load them from file
#         if (os.path.exists(os.path.join(self.path, 'general_stats.csv'))) & (self.force_compute == False):
#             print('Loading general stats from '+os.path.join(self.path, 'general_stats.csv'))
#             general_stats = pd.read_csv(os.path.join(self.path, 'general_stats.csv'), index_col = 0)
#         # If the stats are not computed compute them and save them in file
#         else:
#             print('Computing general stats and saving to '+os.path.join(self.path, 'general_stats.csv'))
#             # Define auxiliary tcga dataframe to obtain healthy tcga samples
#             tcga_df = self.label_df[self.label_df['is_tcga']]

#             # Get the identifiers of the samples in each subset
#             gtex_samples = self.label_df[self.label_df['is_tcga']==False].index
#             tcga_samples = tcga_df.index
#             healthy_tcga_samples = tcga_df[tcga_df['lab_txt'].str.contains('GTEX')].index
#             joint_samples = self.label_df.index

#             # Compute the mean of the subsets
#             tqdm.pandas(desc="Computing Mean GTEx")
#             gtex_mean = self.matrix_data.loc[:, gtex_samples].progress_apply(np.mean, axis = 1).to_frame(name='gtex_mean')
#             tqdm.pandas(desc="Computing Mean TCGA")
#             tcga_mean = self.matrix_data.loc[:, tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='tcga_mean')
#             tqdm.pandas(desc="Computing Mean Healthy TCGA")
#             healthy_tcga_mean = self.matrix_data.loc[:, healthy_tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='healthy_tcga_mean')
#             tqdm.pandas(desc="Computing Joint Mean")
#             joint_mean = self.matrix_data.loc[:, joint_samples].progress_apply(np.mean, axis = 1).to_frame(name='joint_mean')

#             # Compute the std of the subsets
#             tqdm.pandas(desc="Computing std GTEx")
#             gtex_std = self.matrix_data.loc[:, gtex_samples].progress_apply(np.std, axis = 1).to_frame(name='gtex_std')
#             tqdm.pandas(desc="Computing std TCGA")
#             tcga_std = self.matrix_data.loc[:, tcga_samples].progress_apply(np.std, axis = 1).to_frame(name='tcga_std')
#             tqdm.pandas(desc="Computing std Healthy TCGA")
#             healthy_tcga_std = self.matrix_data.loc[:, healthy_tcga_samples].progress_apply(np.std, axis = 1).to_frame(name='healthy_tcga_std')
#             tqdm.pandas(desc="Computing Joint std")
#             joint_std = self.matrix_data.loc[:, joint_samples].progress_apply(np.std, axis = 1).to_frame(name='joint_std')

#             # Compute the fraction of samples where a gene is expressed
#             print('Computing fraction of samples where each gene is expressed ...')
#             min_val = self.matrix_data.min().min() # Get minimum value
#             tqdm.pandas(desc="Computing Expressed Genes")
#             expressed_matrix = self.matrix_data.progress_apply(lambda x: x>min_val, axis = 1) # Compute expressed positions
            
#             # Compute expressed sample fractions for all subsets
#             tqdm.pandas(desc="Computing Joint Expressed Sample Fraction")
#             joint_sample_fraction = expressed_matrix.progress_apply(np.mean, axis = 1).to_frame(name='joint_sample_frac')
#             tqdm.pandas(desc="Computing GTEx Expressed Sample Fraction")
#             gtex_sample_fraction = expressed_matrix.loc[:, gtex_samples].progress_apply(np.mean, axis = 1).to_frame(name='gtex_sample_frac')
#             tqdm.pandas(desc="Computing TCGA Expressed Sample Fraction")
#             tcga_sample_fraction = expressed_matrix.loc[:, tcga_samples].progress_apply(np.mean, axis = 1).to_frame(name='tcga_sample_frac')
            

#             # Join stats in single dataframe
#             general_stats = pd.concat([gtex_mean, tcga_mean, healthy_tcga_mean, joint_mean,
#                                         gtex_std, tcga_std, healthy_tcga_std, joint_std,
#                                         joint_sample_fraction, gtex_sample_fraction, tcga_sample_fraction,], axis=1)
#             general_stats.to_csv(os.path.join(self.path, 'general_stats.csv'))

#         return general_stats

#     # This function filters out genes by mean, standard deviation, expression fraction, random fraction or list of genes
#     def filter_genes(self):
#         # If there is a gene list specified by parameter then it overwrites mean, std and rand_frac filtering  
#         if self.gene_list_csv != 'None':
#             # Print user message
#             print(f'CanDLE will train with the list of genes specified in {self.gene_list_csv}')
#             gene_csv_df = pd.read_csv(self.gene_list_csv, index_col=0)
#             gene_list = pd.Index(gene_csv_df['gene_name'])
        
#         # If no list of genes is specified then proceed with mean, std, sample_frac and rand_frac filtering
#         elif (not os.path.exists(os.path.join(self.dataset_info_path, "filtering_info.csv"))) or self.force_compute:
            
#             print("Computing list of filtered genes. And saving filtering info to:\n\t"+ os.path.join(self.dataset_info_path, "filtering_info.csv"))
            
#             # Find the indices of the samples with mean, standard deviation and sample fractions that fulfill the thresholds
#             mean_bool_index = ((self.general_stats['joint_mean']>self.mean_thr) & (self.general_stats['gtex_mean']>self.mean_thr) & (self.general_stats['tcga_mean']>self.mean_thr))
#             std_bool_index = ((self.general_stats['joint_std']>self.std_thr) & (self.general_stats['gtex_std']>self.std_thr) & (self.general_stats['tcga_std']>self.std_thr))
#             sample_frac_bool_index = ((self.general_stats['joint_sample_frac'] > self.sample_frac) & (self.general_stats['gtex_sample_frac'] > self.sample_frac) & (self.general_stats['tcga_sample_frac'] > self.sample_frac))
            
#             # Compute intersection of mean_data_index and std_data_index
#             mean_std_sample_index = np.logical_and.reduce((mean_bool_index.values, std_bool_index.values, sample_frac_bool_index)).ravel()
#             # Make a gene list of the samples that fulfill the thresholds
#             gene_list = self.matrix_data.index[mean_std_sample_index]

#             # Subsample gene list in case self.rand_frac < 1
#             if self.rand_frac < 1:
#                 np.random.seed(0) # Ensure reproducibility # TODO: Parametrize this seed to run variation experiments
#                 rand_selector = np.zeros(len(gene_list))
#                 rand_selector[:int(len(gene_list)*self.rand_frac)] = 1
#                 np.random.shuffle(rand_selector) # Shuffle boolean selector
#                 rand_selector = np.array(rand_selector, dtype=bool)
#                 gene_list = gene_list[rand_selector] # Filter gene list based in rand_selector
            
#             # Compute boolean value for each gene that indicates if it was included in the filtered gene list
#             included_in_filtering = self.general_stats.index.isin(gene_list)

#             # Merge all statistics and included_in_filtering into a final dataframe
#             filtering_info_df = self.general_stats
#             filtering_info_df['included'] = included_in_filtering
#             filtering_info_df.index.name = "gene"

#             # Save the filtering info to files
#             filtering_info_df.to_csv(os.path.join(self.dataset_info_path, "filtering_info.csv"), index = True)
#             # Plot histograms with plot_filtering_histograms()
#             self.plot_filtering_histograms(filtering_info_df)

#         else:
#             print("Loading filtering info from:\n\t" + os.path.join(self.dataset_info_path, "filtering_info.csv"))
#             filtering_info_df = pd.read_csv(os.path.join(self.dataset_info_path, "filtering_info.csv"), index_col = 0)
#             # get indices of filtering_info_df that are True in the included column
#             gene_list = filtering_info_df.index[filtering_info_df["included"].values == True]
#             # Plot histograms with plot_filtering_histograms()
#             self.plot_filtering_histograms(filtering_info_df)
        
#         # Filter tha data matrix based on the gene list
#         gene_filtered_data_matrix = self.matrix_data_filtered.loc[gene_list, :]

#         print("Currently working with {} genes...".format(gene_filtered_data_matrix.shape[0]))

#         return gene_list.to_list(), gene_filtered_data_matrix
    
#     # This function performs a data normalization by batches (GTEX or TCGA) 
#     def batch_normalize(self):
#         if self.batch_normalization=='None':
#             print('Did not perform batch normalization...')
#             return
#         else:
#             print('Batch normalizing matrix data...')
#             start = time.time()
#             # Get the identifiers of the samples in each subset
#             gtex_samples = self.label_df[self.label_df['is_tcga']==False].index
#             tcga_samples = self.label_df[self.label_df['is_tcga']==True].index
#             # Get stats of the valid genes
#             valid_stats = self.general_stats.loc[self.filtered_gene_list, :]

#             # Transforms GTEx data
#             normalized_gtex = self.gene_filtered_data_matrix[gtex_samples].sub(valid_stats['gtex_mean'], axis=0)
#             normalized_gtex = normalized_gtex.div(valid_stats['gtex_std'], axis=0)
           
#            # Transform TCGA data according to self.batch_normalization
#             if self.batch_normalization=='normal':
#                 normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['tcga_mean'], axis=0)
#                 normalized_tcga = normalized_tcga.div(valid_stats['tcga_std'], axis=0)
            
#             elif self.batch_normalization=='healthy_tcga':
#                 normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['healthy_tcga_mean'], axis=0)
#                 normalized_tcga = normalized_tcga.div(valid_stats['healthy_tcga_std'], axis=0)
            
#             else:
#                 raise ValueError('Batch normalization should be None, normal or healthy_tcga.')

#             normalized_joint = pd.concat([normalized_gtex, normalized_tcga], axis=1)

#             # Sort columns of normalized joint
#             normalized_joint = normalized_joint.T.sort_index().T

#             # Replace NaNs generated by std division to 0's
#             self.gene_filtered_data_matrix = normalized_joint.fillna(0.0)
#             end = time.time()
#             print(f'It took {round(end-start, 2)} s to batch normalize the data.')

#     # This function uses self.binary_dict to modify self.label_df and self.lab_txt_2_lab_num to make the labels binary
#     def make_binary_problem(self):
#         # If binary_dict is not specified, do not make binary
#         if self.binary_dict == {}:
#             print("No binary problem specified.")
#             return
#         # If binary_dict is specified, make binary
#         else:
#             self.lab_txt_2_lab_num = self.binary_dict
#             # Define numeric labels from the textual labels in self.label_df
#             self.label_df["lab_num"] = self.label_df["lab_txt"].map(self.lab_txt_2_lab_num)   
#             print("Made binary problem.")
#             print(f"Number of samples in class 0: {len(self.label_df[self.label_df['lab_num'] == 0])}, number of samples in class 1: {len(self.label_df[self.label_df['lab_num'] == 1])}")
#             return

#     # This function uses self.label_df to split the data into train, validation and test sets
#     def split_data(self):
#         train_val_lab, test_lab = train_test_split(self.label_df["lab_num"], test_size = 0.2, random_state = self.partition_seed, stratify = self.label_df["lab_num"].values)
#         train_lab, val_lab = train_test_split(train_val_lab, test_size = 0.25, random_state = self.partition_seed, stratify = train_val_lab.values)
#         # Use label indexes to subset the data in self.matrix_data_filtered
#         train_matrix = self.gene_filtered_data_matrix[train_lab.index]
#         val_matrix = self.gene_filtered_data_matrix[val_lab.index]
#         test_matrix = self.gene_filtered_data_matrix[test_lab.index]
#         train_val_matrix = self.gene_filtered_data_matrix[train_val_lab.index]
#         # Declare label dictionaries
#         split_labels = {"train": train_lab, "val": val_lab, "test": test_lab, "train_val": train_val_lab}
#         # Declare matrix dictionaries
#         split_matrices = {"train": train_matrix, "val": val_matrix, "test": test_matrix, "train_val": train_val_matrix}
#         # Both matrixes and labels are already shuffled
#         return split_labels, split_matrices

#     # This function gets the dataloaders for the train, val and test sets
#     def get_dataloaders(self, batch_size):
#         # Select data partitions
#         # These data matrices have samples in rows and genes in columns
#         x_train = torch.Tensor(self.split_matrices["train"].T.values).type(torch.float)
#         x_val = torch.Tensor(self.split_matrices["val"].T.values).type(torch.float)
#         x_test = torch.Tensor(self.split_matrices["test"].T.values).type(torch.float)
        
#         # Cast labels as tensors
#         y_train = torch.Tensor(self.split_labels["train"].values).type(torch.long)
#         y_val = torch.Tensor(self.split_labels["val"].values).type(torch.long)
#         y_test = torch.Tensor(self.split_labels["test"].values).type(torch.long)

#         # Define train, val and test datasets
#         train_dataset = TensorDataset(x_train, y_train)
#         val_dataset = TensorDataset(x_val, y_val)
#         test_dataset = TensorDataset(x_test, y_test)

#         # Create dataloaders
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#         return train_loader, val_loader, test_loader

#     # This function plots a 2X2 figure with the histograms of mean expression and standard deviation before and after filtering
#     def plot_filtering_histograms(self, filtering_info_df):        
#         # Make a figure
#         fig, axes = plt.subplots(2, 2, figsize = (18, 12))
#         fig.suptitle("Filtering of Mean > " +str(self.mean_thr) + " and Standard Deviation > " + str(self.std_thr) , fontsize = 30)
#         # Variable to adjust display height of the histograms
#         max_hist = np.zeros((2, 2))

#         # Plot the histograms of mean and standard deviation before filtering
#         n, _, _ = axes[0, 0].hist(filtering_info_df["joint_mean"], bins = 50, color = "k", density=True)
#         max_hist[0, 0] = np.max(n)
#         axes[0, 0].set_title("Before filtering", fontsize = 26)        
#         n, _, _ = axes[1, 0].hist(filtering_info_df["joint_std"], bins = 50, color = "k", density=True)
#         max_hist[1, 0] = np.max(n)
#         # Plot the histograms of mean and standard deviation after filtering
#         n, _, _ = axes[0, 1].hist(filtering_info_df["joint_mean"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
#         max_hist[0, 1] = np.max(n)
#         axes[0, 1].set_title("After filtering", fontsize = 26)
#         n, _, _ = axes[1, 1].hist(filtering_info_df["joint_std"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
#         max_hist[1, 1] = np.max(n)

#         # Format axes
#         for i in range(2):
#             for j in range(2):
#                 axes[i, j].set_ylabel("Density", fontsize = 16)
#                 axes[i, j].tick_params(labelsize = 14)
#                 axes[i, j].grid(True)
#                 axes[i, j].set_axisbelow(True)
#                 axes[i, j].set_ylim(0, max_hist[i, j] * 1.1)
#                 # Handle mean expression plots
#                 if i == 0:
#                     axes[i, j].set_xlabel("Mean expression", fontsize = 16)
#                     axes[i, j].set_xlim(filtering_info_df["joint_mean"].min(), filtering_info_df["joint_mean"].max())
#                     axes[i, j].plot([self.mean_thr, self.mean_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")
#                 # Handle standard deviation plots
#                 else:
#                     axes[i, j].set_xlabel("Standard deviation", fontsize = 16)
#                     axes[i, j].set_xlim(filtering_info_df["joint_std"].min(), filtering_info_df["joint_std"].max())
#                     axes[i, j].plot([self.std_thr, self.std_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")

#         # Save the figure
#         fig.savefig(os.path.join(self.dataset_info_path, "filtering_histograms.png"), dpi = 300)
#         plt.close(fig)

#     def plot_dim_reduction(self):
#         valid_samples = self.gene_filtered_data_matrix.columns
#         valid_genes = self.filtered_gene_list
#         bool_sample_index = self.matrix_data.columns.isin(valid_samples)
#         bool_gene_index = self.matrix_data.index.isin(valid_genes)
#         print('Filtering raw data to valid genes and samples...')
#         raw_data = self.matrix_data.loc[bool_gene_index, bool_sample_index].T.sort_index()
#         processed_data = self.gene_filtered_data_matrix.T.sort_index()
#         # Get and sort metadata
#         meta_df = self.label_df
#         meta_df = meta_df.sort_index()

#         if (not os.path.exists(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'))) or self.force_compute:
            
#             print(f'Computing dimensionality reduction and saving it to {os.path.join(self.dataset_info_path, "dimensionality_reduction.pkl")}...')
            
#             print('Reducing dimensions with PCA...')
#             raw_pca = PCA(n_components=2, random_state=0)
#             processed_pca = PCA(n_components=2, random_state=0)
#             r_raw_pca = raw_pca.fit_transform(raw_data)
#             r_processed_pca = processed_pca.fit_transform(processed_data)

#             # If this shows a warning with OpenBlast you can solve it using this GitHub issue https://github.com/ultralytics/yolov5/issues/2863
#             # You just need to write in terminal "export OMP_NUM_THREADS=1"
#             print('Reducing dimensions with TSNE...')
#             raw_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='random', n_jobs=-1, verbose=2)       # learning_rate and init were set due to a future warning
#             processed_tsne = TSNE(n_components=2, random_state=0, learning_rate='auto', init='random', n_jobs=-1, verbose=2)
#             r_raw_tsne = raw_tsne.fit_transform(raw_data)
#             r_processed_tsne = processed_tsne.fit_transform(processed_data)

#             print('Reducing dimensions with UMAP...')
#             raw_umap = UMAP(n_components=2, random_state=0) # , n_neighbors=64, local_connectivity=32)          # n_neighbors and local_connectivity are set to ensure that the graph is connected
#             processed_umap = UMAP(n_components=2, random_state=0) # , n_neighbors=64, local_connectivity=32)
#             r_raw_umap = raw_umap.fit_transform(raw_data)
#             r_processed_umap = processed_umap.fit_transform(processed_data)

#             reduced_dict = {'raw_pca': r_raw_pca,               'raw_tsne': r_raw_tsne,                 'raw_umap': r_raw_umap,
#                             'processed_pca': r_processed_pca,   'processed_tsne': r_processed_tsne,     'processed_umap':  r_processed_umap}

#             with open(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'), 'wb') as f:
#                 pkl.dump(reduced_dict, f)
        
#         else:
            
#             print(f'Loading dimensionality reduction from {os.path.join(self.dataset_info_path, "dimensionality_reduction.pkl")}...')
            
#             with open(os.path.join(self.dataset_info_path, 'dimensionality_reduction.pkl'), 'rb') as f:
#                 reduced_dict = pkl.load(f)
        

#         def plot_dim_reduction(reduced_dict, meta_df, color_type='tcga', cmap=None):


#             # Load id_2_tissue mapper from file
#             with open(os.path.join(self.path, "mappers", "id_2_tissue_mapper.json"), "r") as f:
#                 id_2_tissue_mapper = json.load(f)

#             meta_df['tissue'] = meta_df['lab_txt'].map(id_2_tissue_mapper)

#             # Get dictionaries to have different options to colorize the scatter points
#             tcga_dict = {True: 1, False: 0}
#             tissue_dict = {tissue: i/len(meta_df.tissue.unique()) for i, tissue in enumerate(sorted(meta_df.tissue.unique()))}
#             class_dict = {cl: i/len(meta_df.lab_txt.unique()) for i, cl in enumerate(sorted(meta_df.lab_txt.unique()))}

#             # Define color map
#             if cmap is None:
#                 d_colors = ["black", "darkcyan"]
#                 cmap = LinearSegmentedColormap.from_list("candle_cmap", d_colors)
#             else:
#                 cmap = get_cmap(cmap)
            
#             # Compute color values
#             if color_type == 'tcga':
#                 meta_df['color'] = meta_df['is_tcga'].map(tcga_dict)
#             elif color_type == 'tissue':
#                 meta_df['color'] = meta_df['tissue'].map(tissue_dict)
#             elif color_type == 'class':
#                 meta_df['color'] = meta_df['lab_txt'].map(class_dict)

#             # Plot figure
#             fig, ax = plt.subplots(2, 3)

#             ax[0, 0].scatter(reduced_dict['raw_pca'][:,0],          reduced_dict['raw_pca'][:,1],           c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[1, 0].scatter(reduced_dict['processed_pca'][:,0],    reduced_dict['processed_pca'][:,1],     c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[0, 1].scatter(reduced_dict['raw_tsne'][:,0],         reduced_dict['raw_tsne'][:,1],          c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[1, 1].scatter(reduced_dict['processed_tsne'][:,0],   reduced_dict['processed_tsne'][:,1],    c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[0, 2].scatter(reduced_dict['raw_umap'][:,0],         reduced_dict['raw_umap'][:,1],          c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)
#             ax[1, 2].scatter(reduced_dict['processed_umap'][:,0],   reduced_dict['processed_umap'][:,1],    c = meta_df['color'], s=1, cmap=cmap, vmin=0, vmax=1, alpha=0.05)

#             label_dict = {0: 'PC', 1: 'T-SNE', 2: 'UMAP'}
#             title_dict = {0: 'PCA', 1: 'T-SNE', 2: 'UMAP'}

#             for i in range(ax.shape[0]):
#                 for j in range(ax.shape[1]):
#                     ax[i,j].spines.right.set_visible(False) 
#                     ax[i,j].spines.top.set_visible(False)
#                     ax[i,j].set_xlabel(f'{label_dict[j]}1')
#                     ax[i,j].set_ylabel(f'{label_dict[j]}2')
#                     ax[i,j].set_title(f'{title_dict[j]} of Raw Data' if i==0 else f'{title_dict[j]} of Processed Data')

#             # Set the legend
#             if (color_type == 'tcga'):
#                 handles = [mpatches.Patch(facecolor=cmap(0.0), edgecolor='black', label='GTEx'),
#                            mpatches.Patch(facecolor=cmap(1.0), edgecolor='black', label='TCGA')]
#                 fig.set_size_inches(13, 7)
#                 ax[0, 2].legend(handles=handles, loc='center left',bbox_to_anchor=(1.15, 0.49))
#                 plt.tight_layout()

#             elif (color_type == 'tissue'):
#                 handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'{key} Tissue') for key, val in tissue_dict.items()]
#                 fig.set_size_inches(15, 7)
#                 fig.legend(handles=handles, loc=7)
#                 fig.tight_layout()
#                 fig.subplots_adjust(right=0.83)
                

#             elif (color_type == 'class'):
#                 handles = [mpatches.Patch(facecolor=cmap(val), edgecolor='black', label=f'{key}') for key, val in class_dict.items()]
#                 fig.set_size_inches(18, 8)
#                 fig.legend(handles=handles, loc=7, ncol=2)
#                 fig.tight_layout()
#                 fig.subplots_adjust(right=0.75)

            
#             else:
#                 raise ValueError('Invalid color type...')

#             # Save
#             fig.savefig(os.path.join(self.dataset_info_path, f'dim_reduction_{color_type}.png'), dpi=300)
#             plt.close()

#         # Plot dimensionality reduction with the three different color styles
#         plot_dim_reduction(reduced_dict, meta_df, color_type='tcga')
#         plot_dim_reduction(reduced_dict, meta_df, color_type='tissue', cmap='brg')
#         plot_dim_reduction(reduced_dict, meta_df, color_type='class', cmap='brg')