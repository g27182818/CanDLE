import tqdm
import pandas as pd
import numpy as np
import os
import time
import json
import torch
import networkx as nx
import pickle as pkl
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils import *

pd.options.mode.chained_assignment = None  # default='warn'


class ToilDataset():
    def __init__(self, path, tcga = True, gtex = True, mean_thr=0.5, std_thr=0.5, use_graph=True, corr_thr=0.6, p_thr=0.05,
                label_type = 'phenotype', force_compute = False):
        self.path = path
        self.dataset_info_path = os.path.join(self.path, 'processed_data',
                                              'tcga='+str(tcga)+'_gtex='+str(gtex),
                                              'mean_thr='+str(mean_thr)+'_std_thr='+str(std_thr),
                                              'corr_thr='+str(corr_thr)+'_p_thr='+str(p_thr))
        self.tcga = tcga
        self.gtex = gtex
        self.mean_thr = mean_thr
        self.std_thr = std_thr
        self.use_graph = use_graph
        self.corr_thr = corr_thr
        self.p_thr = p_thr
        self.label_type = label_type # can be 'phenotype' or 'category'
        self.force_compute = force_compute

        # Read data from the Toil data set
        self.matrix_data, self.categories, self.phenotypes = self.read_toil_data()
        # Filter toil datasets to use GTEx, TCGA or both
        self.matrix_data_filtered, self.categories_filtered, self.phenotypes_filtered = self.filter_toil_datasets()
        # Filter genes based on mean and std
        self.filtered_gene_list, self.filtering_info, self.gene_filtered_data_matrix = self.filter_genes()
        # Get labels and label dictionary
        self.label_df, self.lab_txt_2_lab_num = self.find_labels()
        # Split data into train, validation and test sets
        self.split_labels, self.split_matrices = self.split_data() # For split_matrices samples are columns and genes are rows
        # Compute correlation graph if use_graph is True
        self.edge_indices, self.edge_attributes = self.compute_graph() if self.use_graph else (torch.empty((2, 1)), torch.empty((1,)))
    
    def read_toil_data(self):
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
            matrix_data (pd.dataframe): Dataframe of the complete Toil data set obtained from read_toil_data().
            categories (pd.dataframe): Dataframe of categories of the complete Toil data set obtained from read_toil_data().
            phenotypes (pd.dataframe): Dataframe of phenotypes of the complete Toil data set obtained from read_toil_data().
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
            categories_filtered = self.categories[~self.categories["sample"].str.contains("GTEX")]
            phenotypes_filtered = phenotypes_filtered[~self.phenotypes["sample"].str.contains("GTEX")]
        elif ( not self.tcga) and self.gtex:
            print("Using GTEX samples only")
            # Filter out all tcga samples from matrix_data
            matrix_data_filtered = matrix_data_filtered.iloc[:, ~matrix_data_filtered.columns.str.contains("TCGA")]
            categories_filtered = self.categories[~self.categories["sample"].str.contains("TCGA")]
            phenotypes_filtered = phenotypes_filtered[~self.phenotypes["sample"].str.contains("TCGA")]
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
            # TODO: Remove the need to always compute the mean and std
            # Compute the mean and standard deviation of the matrix_data
            tqdm.pandas(desc="Computing Mean expression")
            mean_data = self.matrix_data.progress_apply(np.mean, axis = 1)
            tqdm.pandas(desc="Computing Standard Deviation of expression")
            std_data = self.matrix_data.progress_apply(np.std, axis = 1)
            
            # Find the indices of the samples with mean and standard deviation that fulfill the thresholds
            mean_data_index = mean_data > self.mean_thr
            std_data_index = std_data > self.std_thr
            # Compute intersection of mean_data_index and std_data_index
            mean_std_index = np.logical_and(mean_data_index.values, std_data_index.values)
            # Make a gene list of the samples that fulfill the thresholds
            gene_list = self.matrix_data.index[mean_std_index]
            # Compute boolean value for each gene that indicates if it was included in the filtered gene list
            included_in_filtering = mean_data.index.isin(gene_list)

            # Merge the mean, std and included_in_filtering into a final dataframe
            filtering_info_df = pd.DataFrame({"mean": mean_data, "std": std_data, "included": included_in_filtering})
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
        # Load the mapper from textual labels to numerical labels
        with open(os.path.join(self.path, "mappers", "lab_txt_2_lab_num_mapper.json"), "r") as f:
                lab_txt_2_lab_num = json.load(f)
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
            # Put GTEX textual label in lab_txt column for normal (Healthy) TCGA samples
            label_df.loc[normal_tcga_samples, "lab_txt"] = label_df.loc[normal_tcga_samples, "_primary_site"].map(normal_tcga_2_gtex)

            # Map phenotype detailed category to textual label in label_df for the non normal (Healthy) TCGA samples
            label_df.loc[label_df["lab_txt"] == 0, "lab_txt"] = label_df.loc[label_df["lab_txt"] == 0, "detailed_category"].map(pheno_2_lab_txt)
            # Define numeric labels from the textual labels in label_df
            label_df["lab_num"] = label_df["lab_txt"].map(lab_txt_2_lab_num)
        else:
            raise ValueError("label_type must be 'category' or 'phenotype'")
        return label_df, lab_txt_2_lab_num
    
    # This function uses self.label_df to split the data into train, validation and test sets
    def split_data(self):
        train_val_lab, test_lab = train_test_split(self.label_df["lab_num"], test_size = 0.2, random_state = 0, stratify = self.label_df["lab_num"].values)
        train_lab, val_lab = train_test_split(train_val_lab, test_size = 0.25, random_state = 0, stratify = train_val_lab.values)
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
    
    # This function computes a spearman correlation graph between all the genes in the train and val sets
    def compute_graph(self):
        
        # Handle the case when graph is already computed
        if os.path.exists(os.path.join(self.dataset_info_path, "graph_connectivity.csv")) and (not self.force_compute):
            
            print("Loading graph connectivity file from: \n\t{}".format(os.path.join(self.dataset_info_path, "graph_connectivity.csv")))
            # Read pandas from the csv file
            graph_df = pd.read_csv(os.path.join(self.dataset_info_path, "graph_connectivity.csv"), index_col = None)
            # Extract edge indices and edge attrubutes from the dataframe and pass them to tensor
            edge_indices = torch.Tensor(graph_df[["source", "target"]].values.T).type(torch.long)
            edge_attributes = torch.Tensor(graph_df["weight"].values).type(torch.long)

            print("Loading graph info file from: \n\t{}".format(os.path.join(self.dataset_info_path, "graph_info.json")))
            with open(os.path.join(self.dataset_info_path, "graph_info.json"), "r") as f:
                graph_info = json.load(f)
        else:
            print("Computing graph connectivity...")
            # Transpose train_val matrix
            x = self.split_matrices["train_val"].T
            start = time.time()
            correlation, p_value = spearmanr(x)                                                 # Compute Spearman correlation
            end = time.time()
            print("Time to compute spearmanr: {}".format(round(end - start, 2)))
            valid_corr = np.abs(correlation) > self.corr_thr                                    # Filter based on the correlation threshold
            valid_p_value = p_value < self.p_thr                                                # Filter based in p_value threshold
            adjacency_matrix = np.logical_and(valid_corr, valid_p_value).astype(int)            # Compose both filters
            adjacency_matrix = adjacency_matrix - np.eye(adjacency_matrix.shape[0], dtype=int)  # Discard self loops
            adjacency_sparse = coo_matrix(adjacency_matrix)                                     # Pass to sparse matrix
            adjacency_sparse.eliminate_zeros()                                                  # Delete zeros from representation
            edge_indices, edge_attributes = from_scipy_sparse_matrix(adjacency_sparse)          # Get edges and weights in tensors
            
            # Save edge indexs and attributes to a csv file
            graph_df = pd.DataFrame(edge_indices.T.numpy(), columns=["source", "target"])
            graph_df["weight"] = edge_attributes.numpy()
            # TODO: Also save the gene list with numbers to interpret the graph connectivity csv file
            graph_df.to_csv(os.path.join(self.dataset_info_path, "graph_connectivity.csv"), index=False)

            # Compute graph statistics
            print('Computing graph info...')
            nx_graph = nx.from_scipy_sparse_matrix(adjacency_sparse)
            connected_bool = nx.is_connected(nx_graph)
            length_connected = [len(c) for c in sorted(nx.connected_components(nx_graph), key=len, reverse=False)]
            graph_info ={
                        "Node number" : x.shape[1],
                        "Edge number" : edge_indices.shape[1]//2,
                        "Average degree" : round(edge_indices.shape[1]/x.shape[1], 2),
                        "Connected" : connected_bool,
                        "Biggest Con. Comp." : length_connected[-1]
                        }
            # Save normal_tcga_mapper mappers to file
            with open(os.path.join(self.dataset_info_path, "graph_info.json"), 'w') as f:
                json.dump(graph_info, f, indent=4)

        # Print graph info dictionary
        print("Graph info:")
        for key, value in graph_info.items():
            print("\t{}: {}".format(key, value))
        return edge_indices, edge_attributes

        # TODO: Add a function that handles a possible subsampling of the data

    # This function gets the dataloaders for the train, val and test sets
    def get_dataloaders(self, batch_size):
        # Cast data as tensors
        # These data matrices have samples in rows and genes in columns
        x_train = torch.Tensor(self.split_matrices["train"].T, dtype=torch.float)
        x_val = torch.Tensor(self.split_matrices["val"].T, dtype=torch.float)
        x_test = torch.Tensor(self.split_matrices["test"].T, dtype=torch.float)
        # Cast labels as tensors
        y_train = torch.Tensor(self.split_labels["train"].values, dtype=torch.long)
        y_val = torch.Tensor(self.split_labels["val"].values, dtype=torch.long)
        y_test = torch.Tensor(self.split_labels["test"].values, dtype=torch.long)
        # Define train graph list
        train_graph_list = [Data(x=torch.unsqueeze(x_train[i, :], 1),
                                 y=y_train[i],
                                 edge_index=self.edge_indices,
                                 edge_attr=self.edge_attributes,
                                 num_nodes= len(x_train[i,:])) for i in range(x_train.shape[0])]
        # Define val graph list
        val_graph_list = [Data(x=torch.unsqueeze(x_val[i, :], 1),
                               y=y_val[i],
                               edge_index=self.edge_indices,
                               edge_attr=self.edge_attributes,
                               num_nodes= len(x_val[i,:])) for i in range(x_val.shape[0])]
        # Define test graph list
        test_graph_list = [Data(x=torch.unsqueeze(x_test[i, :], 1),
                                y=y_test[i],
                                edge_index=self.edge_indices,
                                edge_attr=self.edge_attributes,
                                num_nodes= len(x_test[i,:])) for i in range(x_test.shape[0])]

        # Create dataloaders
        train_loader = DataLoader(train_graph_list, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graph_list, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graph_list, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader, test_loader
        

test_toil_dataset = ToilDataset(os.path.join("data", "toil_data"),
                                tcga = True,
                                gtex = True,
                                mean_thr = 3.0,
                                std_thr = 0.5,
                                use_graph = True,
                                corr_thr = 0.6,
                                p_thr = 0.05,
                                label_type = 'phenotype',
                                force_compute = True)


