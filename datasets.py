import subprocess
import tqdm
import pandas as pd
import numpy as np
import os
import time
import json
import torch
import zipfile
import gzip
import shutil
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from utils import *

# Suppress not useful warnings
pd.options.mode.chained_assignment = None  # default='warn'


class ToilDataset():
    def __init__(self, path, dataset = 'both', tissue='all', binary_dict={}, mean_thr=0.5,
                std_thr=0.5, label_type = 'phenotype', batch_normalization='none', partition_seed=0,
                force_compute = False):
        self.path = path
        self.tissue = tissue
        self.binary_dict = binary_dict
        self.tcga = (dataset == 'tcga') or (dataset == 'both')
        self.gtex = (dataset == 'gtex') or (dataset == 'both')
        self.dataset_info_path = os.path.join(self.path, 'processed_data',
                                              'dataset='+str(dataset),
                                              'mean_thr='+str(mean_thr)+'_std_thr='+str(std_thr), 
                                              'tissue='+str(self.tissue))
        self.mean_thr = mean_thr
        self.std_thr = std_thr
        self.label_type = label_type # can be 'phenotype' or 'category'
        self.batch_normalization = batch_normalization # TODO: Just allow batch normalization under certain conditions of datasets
        self.partition_seed = partition_seed # seed for train/val/test split
        self.force_compute = force_compute

        # Main Bioinformatic pipeline

        # Make mapper files if they are not already saved
        self.make_mappers()
        # Read data from the Toil data set
        self.matrix_data, self.categories, self.phenotypes = self.read_data()
        # Filter toil datasets to use GTEx, TCGA or both
        self.matrix_data_filtered, self.categories_filtered, self.phenotypes_filtered = self.filter_toil_datasets()
        # Filter genes based on mean and std
        self.filtered_gene_list, self.filtering_info, self.gene_filtered_data_matrix = self.filter_genes()
        # Get labels and label dictionary. 
        self.label_df, self.lab_txt_2_lab_num = self.find_labels()
        # Find stats of each dataset segment
        self.general_stats = self.find_means_and_stds()
        # Perform batch normalization this uses files saved by self.find_means_and_stds() and normalizes self.gene_filtered_data_matrix  
        self.batch_normalize()
        # Filter self.label_df and self.lab_txt_2_lab_num based on the specified tissue
        self.filter_by_tissue()
        # Make the problem binary in case self.binary_dict is not empty
        self.make_binary_problem() # If self.binary_dict == {} this function does nothing
        # Split data into train, validation and test sets. This function uses self.label_df to split the data with the same proportion.
        self.split_labels, self.split_matrices = self.split_data() # For split_matrices samples are columns and genes are rows

        # Define number of classes for classification
        self.num_classes = len(self.lab_txt_2_lab_num.keys()) if self.binary_dict == {} else 2

        # Make important plots with dataset characteristics
        self.plot_label_distribution()
        # self.plot_gene_expression_histograms(rand_size=100000)

    def make_mappers(self):
        """
        This function generates mapper files useful for class definition in the dataset by running the make_mappers.py file
        """
        # Just make mappers if they are not already saved
        if not os.path.exists(os.path.join(self.path), 'mappers', 'category_mapper.json'):
            # run main.py with subprocess
            command = f'python make_mappers.py'
            print(command)
            command = command.split()
            subprocess.call(command)    

    def read_data(self):
        """
        Reads data from the Toil data set with root path and returns matrix_data, categories and phenotypes dataframes.

        Args:
            path (str): Root path of the Toil data set.

        Returns:
            matrix_data (pd.dataframe): Matrix data of the Toil data set. Columns are samples and rows are genes.
            categories (pd.dataframe): Categories of the Toil data set. Rows are samples.
            phenotypes (pd.dataframe): Detailed phenotypes of the Toil data set. Rows are samples.
        """
        start = time.time()
        matrix_data = pd.read_feather(os.path.join(self.path, "data_matrix.feather"))
        categories = pd.read_csv(os.path.join(self.path, "categories.csv"), encoding = "cp1252")
        phenotypes = pd.read_csv(os.path.join(self.path, "phenotypes.csv"), encoding = "cp1252")
        # Delete the first column of categories and phenotypes
        categories = categories.drop(categories.columns[0], axis = 1)
        phenotypes = phenotypes.drop(phenotypes.columns[0], axis = 1)
        # Set the first column of matrix_data, categories and phenotypes as index
        matrix_data.set_index(matrix_data.columns[0], inplace = True)
        categories.set_index(categories.columns[0], inplace = True)
        phenotypes.set_index(phenotypes.columns[0], inplace = True)
        # Delete the rows with nan values
        categories.dropna(inplace=True)
        phenotypes.dropna(inplace=True)
        end = time.time()
        print("Time to load data: {} s".format(round(end - start, 3)))
        return matrix_data, categories, phenotypes
    
    def filter_toil_datasets(self):
        """
        Filters the Toil data set by using or not using TCGA and GTEx samples.

        Args:
            matrix_data (pd.dataframe): Dataframe of the complete Toil data set obtained from read_data().
            categories (pd.dataframe): Dataframe of categories of the complete Toil data set obtained from read_data().
            phenotypes (pd.dataframe): Dataframe of phenotypes of the complete Toil data set obtained from read_data().
            tcga (bool, optional): If True, TCGA samples are used. Defaults to True. 
            gtex (bool, optional): If True GTEX samples are used. Defaults to True.

        Raises:
            ValueError: If both tcga and gtex are False then an error is raised because there is no data to use.

        Returns:
            matrix_data_filtered(pd.dataframe): Dataframe of the filtered Toil data set.
            categories_filtered(pd.dataframe): Dataframe of the filtered categories of the Toil data set.
            phenotypes_filtered(pd.dataframe): Dataframe of the filtered phenotypes of the Toil data set.
        """
        # Filter out all TARGET samples from matrix_data
        self.matrix_data = self.matrix_data.iloc[:, ~self.matrix_data.columns.str.contains("TARGET")]
        matrix_data_filtered = self.matrix_data
        # Filter out all TARGET samples from self.phenotypes
        phenotypes_filtered = self.phenotypes[~self.phenotypes.index.str.contains("TARGET")]

        # Handle the filters for TCGA and GTEx
        if self.tcga and ( not self.gtex):
            print("Using TCGA samples only")
            # Filter out all gtex samples from matrix_data
            matrix_data_filtered = matrix_data_filtered.iloc[:, ~matrix_data_filtered.columns.str.contains("GTEX")]
            categories_filtered = self.categories.loc[~self.categories.index.str.contains("GTEX"), :]
            phenotypes_filtered = phenotypes_filtered.loc[~(phenotypes_filtered["_study"]=="GTEX"), :]
        elif ( not self.tcga) and self.gtex:
            print("Using GTEX samples only")
            # Filter out all tcga samples from matrix_data
            matrix_data_filtered = matrix_data_filtered.iloc[:, ~matrix_data_filtered.columns.str.contains("TCGA")]
            categories_filtered = self.categories.loc[~self.categories.index.str.contains("TCGA"), :]
            phenotypes_filtered = phenotypes_filtered.loc[~(phenotypes_filtered["_study"]=="TCGA"), :]
        elif self.tcga and self.gtex:
            # Do nothing because both TCGA and GTEX samples are included
            print("Using TCGA and GTEX samples")
            matrix_data_filtered = self.matrix_data
            categories_filtered = self.categories
            phenotypes_filtered = phenotypes_filtered
        else:
            raise ValueError("You are not selecting any dataset.")
            
        return matrix_data_filtered, categories_filtered, phenotypes_filtered
        
    # This function computes the mean and standard deviation of the matrix_data and filters out the samples with mean and standard deviation below the thresholds
    def filter_genes(self):
        if (not os.path.exists(self.dataset_info_path)) or self.force_compute:
            print("Computing mean, std and list of filtered genes. And saving filtering info to:\n\t"+ os.path.join(self.dataset_info_path, "filtering_info.csv"))
            # Make directory if it does not exist
            os.makedirs(self.dataset_info_path, exist_ok = True)
            # Compute the mean and standard deviation of the matrix_data if they do not exist
            if (not os.path.exists(os.path.join(self.path, "mean_expression.csv"))) or (not os.path.exists(os.path.join(self.path, "std_expression.csv"))) or self.force_compute:
                tqdm.pandas(desc="Computing Mean expression")
                mean_data = self.matrix_data.progress_apply(np.mean, axis = 1).to_frame(name='mean')
                tqdm.pandas(desc="Computing Standard Deviation of expression")
                std_data = self.matrix_data.progress_apply(np.std, axis = 1).to_frame(name='std')

                # Save the mean and std to a csv file
                mean_data.to_csv(os.path.join(self.path, "mean_expression.csv"))
                std_data.to_csv(os.path.join(self.path, "std_expression.csv"))
            # Load the mean and std from csv files if they exist
            else:
                mean_data = pd.read_csv(os.path.join(self.path, "mean_expression.csv"), index_col = 0)
                std_data = pd.read_csv(os.path.join(self.path, "std_expression.csv"), index_col = 0)
            
            # Find the indices of the samples with mean and standard deviation that fulfill the thresholds
            mean_data_index = mean_data > self.mean_thr
            std_data_index = std_data > self.std_thr
            # Compute intersection of mean_data_index and std_data_index
            mean_std_index = np.logical_and(mean_data_index.values, std_data_index.values).ravel()
            # Make a gene list of the samples that fulfill the thresholds
            gene_list = self.matrix_data.index[mean_std_index]
            # Compute boolean value for each gene that indicates if it was included in the filtered gene list
            included_in_filtering = mean_data.index.isin(gene_list)

            # Merge the mean, std and included_in_filtering into a final dataframe
            filtering_info_df = pd.DataFrame({"mean": mean_data['mean'], "std": std_data['std'], "included": included_in_filtering})
            filtering_info_df.index.name = "gene"

            # Save the gene list, mean and standard deviation of the matrix_data to files
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

        return gene_list.to_list(), filtering_info_df, gene_filtered_data_matrix

    # This function plots a 2X2 figure with the histograms of mean expression and standard deviation before and after filtering
    def plot_filtering_histograms(self, filtering_info_df):        
        # Make a figure
        fig, axes = plt.subplots(2, 2, figsize = (18, 12))
        fig.suptitle("Filtering of Mean > " +str(self.mean_thr) + " and Standard Deviation > " + str(self.std_thr) , fontsize = 30)
        # Variable to adjust display height of the histograms
        max_hist = np.zeros((2, 2))

        # Plot the histograms of mean and standard deviation before filtering
        n, _, _ = axes[0, 0].hist(filtering_info_df["mean"], bins = 50, color = "k", density=True)
        max_hist[0, 0] = np.max(n)
        axes[0, 0].set_title("Before filtering", fontsize = 26)        
        n, _, _ = axes[1, 0].hist(filtering_info_df["std"], bins = 50, color = "k", density=True)
        max_hist[1, 0] = np.max(n)
        # Plot the histograms of mean and standard deviation after filtering
        n, _, _ = axes[0, 1].hist(filtering_info_df["mean"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
        max_hist[0, 1] = np.max(n)
        axes[0, 1].set_title("After filtering", fontsize = 26)
        n, _, _ = axes[1, 1].hist(filtering_info_df["std"][filtering_info_df["included"]==True], bins = 50, color = "k", density=True)
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
                    axes[i, j].set_xlim(filtering_info_df["mean"].min(), filtering_info_df["mean"].max())
                    axes[i, j].plot([self.mean_thr, self.mean_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")
                # Handle standard deviation plots
                else:
                    axes[i, j].set_xlabel("Standard deviation", fontsize = 16)
                    axes[i, j].set_xlim(filtering_info_df["std"].min(), filtering_info_df["std"].max())
                    axes[i, j].plot([self.std_thr, self.std_thr], [0, 1.2*max_hist[i,j]], color = "r", linestyle = "--")

        # Save the figure
        fig.savefig(os.path.join(self.dataset_info_path, "filtering_histograms.png"), dpi = 300)
        plt.close(fig)

    # This function extracts the labels from categories_filtered or phenotypes_filtered and returns a list of labels and a dictionary of categories to labels
    def find_labels(self):
        # Load mapper dict from normal TCGA samples to GTEX category
        with open(os.path.join(self.path, "mappers", "normal_tcga_2_gtex_mapper.json"), "r") as f:
            normal_tcga_2_gtex = json.load(f)

        if self.label_type == 'category':
            label_df = self.categories_filtered
            
            # Load the categories to textual labels dictionary.
            with open(os.path.join(self.path, "mappers", "category_mapper.json"), "r") as f:
                cat_2_lab_txt = json.load(f)

            # Note: The self.categories_filtered dataframe does not contain any normal (Healthy) TCGA samples.
            # Add one column to the label_df with mapping from TCGA_GTEX_main_category column with cat_2_lab_txt dictionary
            label_df["lab_txt"] = label_df["TCGA_GTEX_main_category"].map(cat_2_lab_txt)
            # Get unique textual labels obtained and sort them
            current_labels = sorted(label_df["lab_txt"].unique().tolist())
            # Define lab_txt_2_lab_num dictionary
            lab_txt_2_lab_num = {lab_txt: i for i, lab_txt in enumerate(current_labels)}

            # Define numeric labels from the textual labels in label_df
            label_df["lab_num"] = label_df["lab_txt"].map(lab_txt_2_lab_num)

        elif self.label_type == 'phenotype':
            label_df = self.phenotypes_filtered

            # Load the phenotypes to textual labels dictionary.
            with open(os.path.join(self.path, "mappers", "phenotype_mapper.json"), "r") as f:
                pheno_2_lab_txt = json.load(f)

            # Declare a new empty column in the label_df for textual labels
            label_df["lab_txt"] = 0
            # Find sample names of normal (Healthy) TCGA subjects
            normal_tcga_samples = self.phenotypes_filtered[self.phenotypes_filtered["_sample_type"] == "Solid Tissue Normal"].index
            
            # Handle normal (Healthy) TCGA subjects
            # If there are both TCGA and GTEX assign GTEX textual label to the normal (Healthy) TCGA subjects
            if self.tcga and self.gtex:
                # Put GTEX textual label in lab_txt column for normal (Healthy) TCGA samples
                label_df.loc[normal_tcga_samples, "lab_txt"] = label_df.loc[normal_tcga_samples, "_primary_site"].map(normal_tcga_2_gtex)
                # pass
            # If there is only TCGA assign TCGA-NT label to the normal (Healthy) TCGA subjects
            elif self.tcga and (not self.gtex):
                # Put TCGA textual label in lab_txt column for normal (Healthy) TCGA samples
                label_df.loc[normal_tcga_samples, "lab_txt"] = "TCGA-NT"
            # If there is GTEX and not TCGA there is no need to handle normal (Healthy) TCGA subjects
            elif self.gtex and (not self.tcga):
                pass
            # If there is neither TCGA nor GTEX raise an error
            else:
                raise ValueError("There is neither TCGA nor GTEX data available.")

            # Map phenotype detailed category to textual label in label_df for the non normal (Healthy) TCGA samples
            label_df.loc[label_df["lab_txt"] == 0, "lab_txt"] = label_df.loc[label_df["lab_txt"] == 0, "detailed_category"].map(pheno_2_lab_txt)
            # Get unique textual labels obtained and sort them
            current_labels = sorted(label_df["lab_txt"].unique().tolist())
            # Define lab_txt_2_lab_num dictionary
            lab_txt_2_lab_num = {lab_txt: i for i, lab_txt in enumerate(current_labels)}
            # Define numeric labels from the textual labels in label_df
            label_df["lab_num"] = label_df["lab_txt"].map(lab_txt_2_lab_num)
        else:
            raise ValueError("label_type must be 'category' or 'phenotype'")
        
        # Save lab_txt_2_lab_num dictionary to json file
        with open(os.path.join(self.dataset_info_path, "lab_txt_2_lab_num_mapper.json"), "w") as f:
            json.dump(lab_txt_2_lab_num, f, indent = 4)
        return label_df, lab_txt_2_lab_num
    
    # This function find the mean expression and std for GTEx, TCGA, healthy TCGA and the joint dataset
    def find_means_and_stds(self):
        # If the info stats are already computed load them from file
        if os.path.exists(os.path.join(self.path, 'general_stats.csv')):
            print('Loading general stats from '+os.path.join(self.path, 'general_stats.csv'))
            general_stats = pd.read_csv(os.path.join(self.path, 'general_stats.csv'), index_col = 0)
        # If the stats are not computed compiute them and save them in file
        else:
            print('Computing general stats and saving to '+os.path.join(self.path, 'general_stats.csv'))
            # Define auxiliary tcga dataframe to obtain healthy tcga samples
            tcga_df = self.label_df[self.label_df['_study']=='TCGA']

            # Get the identidiers of the samples in each subset
            gtex_samples = self.label_df[self.label_df['_study']=='GTEX'].index
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

            # Join stats in single dataframe
            general_stats = pd.concat([gtex_mean, tcga_mean, healthy_tcga_mean, joint_mean, gtex_std, tcga_std, healthy_tcga_std, joint_std], axis=1)
            general_stats.to_csv(os.path.join(self.path, 'general_stats.csv'))

            return general_stats

        # Make histogram plot
        log_bool = False
        density_bool = False
        alpha_hist = 0.7
        colors = [plt.cm.magma(0.8), plt.cm.magma(0.6), plt.cm.magma(0.4), plt.cm.magma(0.2)]
        plt.rcParams['axes.axisbelow'] = True
        plt.figure(figsize=(16, 6))
        plt.subplot(1,2,1)
        general_stats['joint_mean'].hist(bins=100, density=density_bool, log=log_bool, color=colors[0], alpha=alpha_hist, histtype='stepfilled')
        general_stats['gtex_mean'].hist(bins=100, density=density_bool, log=log_bool, color=colors[1], alpha=alpha_hist, histtype='stepfilled')
        general_stats['tcga_mean'].hist(bins=100, density=density_bool, log=log_bool, color=colors[2], alpha=alpha_hist, histtype='stepfilled')
        general_stats['healthy_tcga_mean'].hist(bins=100, density=density_bool, log=log_bool, color=colors[3], alpha=alpha_hist, histtype='stepfilled')
        plt.title('Mean Expression', fontsize=24)
        plt.xlabel('Mean Gene Expression $[\log_2(TPM+0.001)]$', fontsize=18)
        plt.ylabel('Frecuency', fontsize=18)
        plt.legend(['Joint', 'GTEx', 'TCGA', 'Healthy TCGA'], fontsize=12)
        plt.xlim([-10,10])
        plt.subplot(1,2,2)
        general_stats['joint_std'].hist(bins=100, density=density_bool, log=log_bool, color=colors[0], alpha=alpha_hist, histtype='stepfilled')
        general_stats['gtex_std'].hist(bins=100, density=density_bool, log=log_bool, color=colors[1], alpha=alpha_hist, histtype='stepfilled')
        general_stats['tcga_std'].hist(bins=100, density=density_bool, log=log_bool, color=colors[2], alpha=alpha_hist, histtype='stepfilled')
        general_stats['healthy_tcga_std'].hist(bins=100, density=density_bool, log=log_bool, color=colors[3], alpha=alpha_hist, histtype='stepfilled')
        plt.title('Standard Deviation of Expression', fontsize=24)
        plt.xlabel('Std Gene Expression', fontsize=18)
        plt.ylabel('Frecuency', fontsize=18)
        plt.legend(['Joint', 'GTEx', 'TCGA', 'Healthy TCGA'], fontsize=12)
        plt.xlim([0,6])
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'subset_histograms.png'), dpi=300)

        return general_stats

    # This function performs a data normalization 
    def batch_normalize(self):
        if self.batch_normalization=='none':
            print('Did not perform batch normalization...')
            return
        else:
            print('Batch normalizing matrix data...')
            # Define auxiliary tcga dataframe to obtain healthy tcga samples
            tcga_df = self.label_df[self.label_df['_study']=='TCGA']
            # Get the identifiers of the samples in each subset
            gtex_samples = self.label_df[self.label_df['_study']=='GTEX'].index
            tcga_samples = tcga_df.index
            # Get stats of the valid genes
            valid_stats = self.general_stats.loc[self.filtered_gene_list, :]

            # Transforms GTEx data
            normalized_gtex = self.gene_filtered_data_matrix[gtex_samples].sub(valid_stats['gtex_mean'], axis=0)
            normalized_gtex = normalized_gtex.div(valid_stats['gtex_std'], axis=0)
            # This ensures the numerical stability of normalization zeroing genes with std bellow 10^(-8).
            normalized_gtex[normalized_gtex.T.mean()>1e-8] = 0.0 
           
           # Transform TCGA data according to self.batch_normalization
            if self.batch_normalization=='normal':
                normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['tcga_mean'], axis=0)
                normalized_tcga = normalized_tcga.div(valid_stats['tcga_std'], axis=0)
                # This ensures the numerical stability of normalization zeroing genes with std bellow 10^(-8).
                normalized_tcga[normalized_tcga.T.mean()>1e-8] = 0.0 
            
            elif self.batch_normalization=='healthy_tcga':
                normalized_tcga = self.gene_filtered_data_matrix[tcga_samples].sub(valid_stats['healthy_tcga_mean'], axis=0)
                normalized_tcga = normalized_tcga.div(valid_stats['healthy_tcga_std'], axis=0)
                # This ensures the numerical stability of normalization zeroing genes with std bellow 10^(-8).
                normalized_tcga[normalized_tcga.T.mean()>1e-8] = 0.0
            
            else:
                raise ValueError('Batch normalization should be none, normal or healthy_tcga.')

            normalized_joint = pd.concat([normalized_gtex, normalized_tcga], axis=1)
            # Replace Nans generated by std division to 0's
            self.gene_filtered_data_matrix = normalized_joint.fillna(0.0)
        

    # This function modifies self.label_df and self.lab_txt_2_lab_num filtering by the specified tissue in self.tissue
    def filter_by_tissue(self):
        # If tissue is not specified, do not filter
        if self.tissue == 'all':
            print("No Filtering by tissue using all samples.")
            print("Number of samples after filtering by tissue: {}".format(len(self.label_df)))
            return
        # If tissue is specified, filter label_df and lab_txt_2_lab_num
        else:
            # Load id_2_tissue mapper from file
            with open(os.path.join(self.path, "mappers", "id_2_tissue_mapper.json"), "r") as f:
                id_2_tissue_mapper = json.load(f)
            # Handle tha case where tissue is not in mapper
            if self.tissue not in id_2_tissue_mapper.values():
                raise ValueError("Tissue {} is not in the tissue mapper.".format(self.tissue))
            # Define new column in label_df with tissue labels
            self.label_df["tissue"] = self.label_df["lab_txt"].map(id_2_tissue_mapper)
            # Filter label_df by tissue
            self.label_df = self.label_df[self.label_df["tissue"] == self.tissue]
            # Re define current labels
            current_labels = sorted(self.label_df["lab_txt"].unique().tolist())
            # Re compute lab_txt_2_lab_num dictionary
            self.lab_txt_2_lab_num = {lab_txt: i for i, lab_txt in enumerate(current_labels)}
            # Define numeric labels from the textual labels in self.label_df
            self.label_df["lab_num"] = self.label_df["lab_txt"].map(self.lab_txt_2_lab_num)
        
        print("Filtered by {} tissue".format(self.tissue))
        print("Number of samples after filtering by tissue: {}".format(len(self.label_df)))

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
            print("Number of samples in class 0: {}, number of samples in class 1: {}".format(len(self.label_df[self.label_df["lab_num"] == 0]), len(self.label_df[self.label_df["lab_num"] == 1])))
            return

    # This function uses self.label_df to split the data into train, validation and test sets
    def split_data(self):
        train_val_lab, test_lab = train_test_split(self.label_df["lab_num"], test_size = 0.2, random_state = self.partition_seed, stratify = self.label_df["lab_num"].values)
        train_lab, val_lab = train_test_split(train_val_lab, test_size = 0.25, random_state = self.partition_seed, stratify = train_val_lab.values)
        # Use label indexes to subset the data in self.matrix_data_filtered
        train_matrix = self.gene_filtered_data_matrix[train_lab.index]
        val_matrix = self.gene_filtered_data_matrix[val_lab.index]
        test_matrix = self.gene_filtered_data_matrix[test_lab.index]
        train_val_matrix = self.gene_filtered_data_matrix[train_val_lab.index]
        # Declare label dictionaries
        split_labels = {"train": train_lab, "val": val_lab, "test": test_lab, "train_val": train_val_lab}
        # Declare matrix dictionaries
        split_matrices = {"train": train_matrix, "val": val_matrix, "test": test_matrix, "train_val": train_val_matrix}
        # Both matrixes and labels are already shuffled
        return split_labels, split_matrices

    # TODO: Add a function that handles a possible subsampling of the data

    # This function gets the dataloaders for the train, val and test sets
    def get_dataloaders(self, batch_size):
        # Select data partitions
        # These data matrices have samples in rows and genes in columns
        x_train = torch.Tensor(self.split_matrices["train"].T.values).type(torch.float)
        x_val = torch.Tensor(self.split_matrices["val"].T.values).type(torch.float)
        x_test = torch.Tensor(self.split_matrices["test"].T.values).type(torch.float)
        
        # Cast labels as tensors
        y_train = torch.Tensor(self.split_labels["train"].values).type(torch.long)
        y_val = torch.Tensor(self.split_labels["val"].values).type(torch.long)
        y_test = torch.Tensor(self.split_labels["test"].values).type(torch.long)

        # Define train, val and test datasets
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        test_dataset = TensorDataset(x_test, y_test)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, test_loader
    
    # This function plots the label distribution of the dataset
    def plot_label_distribution(self):
        # Reverse salf.lab_txt_2_lab_num dictionary
        lab_num_2_lab_txt = {v: k for k, v in self.lab_txt_2_lab_num.items()}

        # Get label distribution
        train_label_dist = self.split_labels["train"].value_counts()
        val_label_dist = self.split_labels["val"].value_counts()
        test_label_dist = self.split_labels["test"].value_counts()

        # Give distribution textual label indexes
        train_label_dist.index = train_label_dist.index.map(lab_num_2_lab_txt)
        val_label_dist.index = val_label_dist.index.map(lab_num_2_lab_txt)
        test_label_dist.index = test_label_dist.index.map(lab_num_2_lab_txt)
        
        # Handle different fig sizes for gtex and tcga
        if self.gtex and self.tcga:
            fig_size = (15, 15)
        elif self.gtex and not self.tcga:
            fig_size = (15, 7)
        elif not self.gtex and self.tcga:
            fig_size = (15, 7)
        else:
            raise ValueError("Either GTEx or TCGA must be True")


        # Plot horizontal bar chart of label distribution
        plt.figure(figsize=fig_size)
        ax = plt.subplot(1, 3, 1)
        train_label_dist.plot(kind="barh", color="blue", alpha=0.7)
        plt.title("Train")
        plt.xlabel("Label count")
        plt.ylabel("Label")
        plt.yticks(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax = plt.subplot(1, 3, 2)
        val_label_dist.plot(kind="barh", color="green", alpha=0.7)
        plt.title("Validation")
        plt.xlabel("Label count")
        plt.ylabel("Label")
        plt.yticks(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax = plt.subplot(1, 3, 3)
        test_label_dist.plot(kind="barh", color="red", alpha=0.7)
        plt.title("Test")
        plt.xlabel("Label count")
        plt.ylabel("Label")
        plt.yticks(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.tight_layout()

        plt.show()
        plt.savefig(os.path.join(self.dataset_info_path, "label_distribution.png"), dpi=300)
        plt.close()
    
    # This funtion makes histograms of the gene expression values for each dataset Gtex and TCGA
    def plot_gene_expression_histograms(self, rand_size=10000):
        # Get just GTEx  and just TCGA matrices
        gtex_matrix = self.matrix_data.iloc[:, self.matrix_data.columns.str.contains("GTEX")]
        tcga_matrix = self.matrix_data.iloc[:, self.matrix_data.columns.str.contains("TCGA")]

        print("Started sampling gene expression values...")
        np.random.seed(0)
        start = time.time()
        gtex_random_sample = np.random.choice(np.ravel(gtex_matrix.values), size=rand_size)
        tcga_random_sample = np.random.choice(np.ravel(tcga_matrix.values), size=rand_size)
        end = time.time()
        print("Time to sample: {}".format(round(end - start, 3)))
        plt.figure(figsize=(9, 7))
        # Get gene expression histograms
        plt.hist(gtex_random_sample, bins=100, color="blue", alpha=0.7, label="GTEx", log=True, density=True)
        plt.hist(tcga_random_sample, bins=100, color="red", alpha=0.7, label="TCGA", log=True, density=True)
        plt.grid()
        plt.gca().set_axisbelow(True)
        plt.xlabel("Gene Expression $[\log_2(TPM+0.001)]$", fontsize=18)
        plt.ylabel("Density", fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title("Gene Expression Histograms ("+str(rand_size)+" samples)", fontsize=20)
        plt.legend(["GTEx", "TCGA"], fontsize=14)
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(self.dataset_info_path, "gene_expression_histograms.png"), dpi=300)
        plt.close()

        




# Test code for dataset declaration

#test_toil_dataset = ToilDataset(os.path.join("data", "toil_data"),
#                                 dataset = 'both', 
#                                 tissue='all', 
#                                 mean_thr=-10, 
#                                 std_thr=-1.0, 
#                                 label_type = 'phenotype',
#                                 batch_normalization='normal', # Can be 'none', 'normal', 'healthy_tcga'
#                                 partition_seed=0,
#                                 force_compute = False)

#train_loader, val_loader, test_loader = test_toil_dataset.get_dataloaders(batch_size = 100)

#breakpoint()

class WangDataset():
    def __init__(self, path, dataset = 'both', tissue='all', binary_dict={}, mean_thr=0.5,
                std_thr=0.5, partition_seed=0, force_compute = False):

        self.path = path
        self.tissue = tissue
        self.binary_dict = binary_dict
        self.tcga = (dataset == 'tcga') or (dataset == 'both')
        self.gtex = (dataset == 'gtex') or (dataset == 'both')
        self.dataset_info_path = os.path.join(self.path, 'processed_data',
                                              'dataset='+str(dataset),
                                              'tissue='+str(self.tissue))
        self.mean_thr = mean_thr
        self.std_thr = std_thr
        self.partition_seed = partition_seed # seed for train/val/test split
        self.force_compute = force_compute

        # Main Bioinformatic pipeline
        # Make mapper files if they are not already saved
        self.make_mappers()
        # Un-compress data
        self.unzip_data()
        # Read data from the Toil data set
        self.matrix_data, self.categories = self.read_data()

    def make_mappers(self):
        """
        This function generates mapper files useful for class definition in the dataset by running the make_mappers.py file
        """
        # Just make mappers if they are not already saved
        if not os.path.exists(os.path.join(self.path), 'mappers', 'normal_tcga_2_gtex_mapper.json'):
            # run main.py with subprocess
            command = f'python make_mappers.py'
            print(command)
            command = command.split()
            subprocess.call(command) 

    # This function unzips the raw downloaded data from 
    def unzip_data(self):
        final_data_path = os.path.join(self.path, 'original_data')
        # Do nothing if unziped folder already exists
        if os.path.exists(final_data_path):
            print('Files already unzipped...')
            return
        # Unzip data if original_data does not exist
        else:
            print('Unzipping files this may take some minutes...')
            zipped_path = os.path.join(self.path, 'raw_data.zip')
            unzipped_folder = os.path.join(self.path, 'raw_data_unzipped')
            final_data_path = os.path.join(self.path, 'original_data')

            with zipfile.ZipFile(zipped_path, 'r') as zip_ref:
                zip_ref.extractall(unzipped_folder)
            
            classes_paths = os.listdir(unzipped_folder)
            
            # Make final directory
            os.mkdir(final_data_path)
            # Cycle to unzip original data
            for i in tqdm(range(len(classes_paths))):
                class_path = classes_paths[i]
                final_file_name = class_path[:-3]
                # Exclude chol category wich is said to be discarted in original paper but is in the downloaded data
                if not(class_path[:4] == 'chol'):
                #if True:
                    with gzip.open(os.path.join(unzipped_folder, class_path), 'rb') as f_in:
                        with open(os.path.join(final_data_path, final_file_name), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
            
            # Remove temporal folder
            shutil.rmtree(unzipped_folder)
    
    # This helper function recieves the file name of a class and returns a valid textual label
    def get_label_from_name(self, name):
        str_list = name[:-4].split('-')
        if len(str_list) == 5:
            label = str_list[-2]+'-'+str_list[-1]+'-'+str_list[0]
        elif len(str_list) == 4:
            label = str_list[-1]+'-'+str_list[0]
        else:
            raise ValueError('The name of the original file is not adecuate.')
        label = label.upper()

        # TODO: Use a mapper to pass to standard classes
        return label

    # This function reads the data
    def read_data(self):
        # If processed data directory does not exist read, merge and save complete data
        if not os.path.exists(os.path.join(self.path, 'processed_data')):
            data_path = os.path.join(self.path, 'original_data')
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

                act_category = self.get_label_from_name(class_file) # Get label names from file names
                act_category_df = pd.DataFrame({'sample':act_df.columns, 'lab_txt': act_category})
                
                # Join iteratevily data matrices
                data_matrix = act_df if i==0 else data_matrix.join(act_df)
                data_matrix = data_matrix.loc[valid_gene_index, :] # Ensure datamatrix just has common genes in all classes
                category_df = act_category_df if i==0 else pd.concat([category_df, act_category_df], axis=0) # Join category dataframes

            # Put gene at the begining column and resetting index to save in feather
            data_matrix.insert(loc=0, column='Gene_Hugo_Symbol', value=data_matrix.index)
            data_matrix = data_matrix.reset_index()
            # Add a binary column to category_df indicating if the samples is from the TCGA
            category_df['is_tcga'] = category_df['lab_txt'].str.contains('TCGA')
            category_df = category_df.reset_index() # Reset index

            os.mkdir(os.path.join(self.path, 'processed_data'))
            data_matrix.to_feather(os.path.join(self.path, 'processed_data', 'data_matrix.feather'))
            category_df.to_csv(os.path.join(self.path, 'processed_data', 'data_category.csv'))
        # If the data is already merged and stored load it from file
        else:
            start = time.time()
            data_matrix = pd.read_feather(os.path.join(self.path, 'processed_data', 'data_matrix.feather'))
            category_df = pd.read_csv(os.path.join(self.path, 'processed_data', 'data_category.csv'))
            end = time.time()
            del category_df['Unnamed: 0']
            del category_df['index']
            print('Time to read data: {} s'.format(round(end-start,2)))
        
        return data_matrix, category_df


# test_wang = WangDataset(os.path.join('data', 'wang_data'))