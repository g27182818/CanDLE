import os
import glob
import gzip
import shutil
from tcga_downloader import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import pdb
import dask
from dask import dataframe as dd
import dask.array as da
import pickle
import timeit
from dask.diagnostics import ProgressBar
import pickle
import scipy.io as sio
from scipy.stats import spearmanr
from scipy.sparse import coo_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import matplotlib.pyplot as plt
tqdm.pandas()

"""
The mode parameter establishes the functionality of this code.

mode == "download": It downloads the txt files associated to the manifests files (located in the "manifests" directory)
from the TCGA platform. It also unzips this files. This .txt files contain the gene expression vectors and are separated
by cancer type in the "data" directory created.

mode == "threshold": It determines the genes that will be taken into account for model training based on a mean and a
standard deviation threshold. Based on this genes, it will create matrices associated to the gene expression vectors
and labels separated by each type of cancer in the "matrices" directory created.

mode == "go": Will perform the two previous processes sequentially.

There are other possible mode possible values associated to each modular part of this code. Check each block of code if
you wish to run a module during debugging for changing code functionality.

"""

mode = "run_main"
# mode = "threshold"

"""
Definition of thresholds for which useful genes will be determined.
"""

mean_thr = 0.5
std_thr = 0.8

"""
Dictionary that relates cancer type to annotation.
"""

class2anot_dict = {"ACC":1,
                    "BLCA":2,
                    "BRCA":3,
                    "CESC":4,
                    "CHOL":5,
                    "COAD":6,
                    "DLBC":7,
                    "ESCA":8,
                    "GBM":9,
                    "HNSC":10,
                    "KICH":11,
                    "KIRC":12,
                    "KIRP":13,
                    "LAML":14,
                    "LGG":15,
                    "LIHC":16,
                    "LUAD":17,
                    "LUSC":18,
                    "MESO":19,
                    "OV":20,
                    "PAAD":21,
                    "PCPG":22,
                    "PRAD":23,
                    "READ":24,
                    "SARC":25,
                    "SKCM":26,
                    "STAD":27,
                    "TGCT":28,
                    "THCA":29,
                    "THYM":30,
                    "UCEC":31,
                    "UCS":32,
                    "UVM":33}

"""
Code in charge of downloading .txt.gz files with gene expression vectors to "data" directory from TCGA platform based
on manifests
"""

if mode == "download" or mode == "go" or mode == "from_manifests":
    for c in class2anot_dict.keys():
        class_path = os.path.join("manifests", c)
        manifest_paths = glob.glob(class_path+"/*.txt")
        try:
            os.makedirs(os.path.join("data", c))
            print(c + " data directory created")
        except:
            print(c + " data directory already exist")
        ids_tumor = get_ids(os.path.join("manifests", c, "tumor.txt"))
        payload_tumor = prepare_payload(ids_tumor, data_type='Gene Expression Quantification')
        metadata_tumor = get_metadata(payload_tumor)
        print("Starting to download tumor data... ")
        download_data(metadata_tumor, sep="\t", outdir=os.path.join("data", c))

        if len(manifest_paths) > 1:
            ids_normal = get_ids(os.path.join("manifests", c, "normal.txt"))
            payload_normal = prepare_payload(ids_normal, data_type='Gene Expression Quantification')
            metadata_normal = get_metadata(payload_normal)
            print("Starting to download normal data... ")
            download_data(metadata_normal, sep="\t", outdir=os.path.join("data", c))

"""
Code in charge of unzipping .txt.gz files.
"""

if mode == "download" or mode == "go" or mode == "unzip":
    compressed_files = glob.glob("data/*/*/*.txt.gz")
    for compressed_file in compressed_files:
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(compressed_file[:-12]+".txt", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(compressed_file)
        print(compressed_file + " unzipped")

"""
Code in charge of creating a Series of useful genes based on mean and standard deviation thresholds. It also creates
a loc2gene dictionary which maps the gene position in the expression vector to the name of the represented gene.
"""

if mode == "threshold" or mode == "go" or mode == "useful_genes":
    paths = glob.glob("data/*/*/*.txt")
    print("Creating complete dataframe...")
    complete_df = dd.concat([dd.read_csv(path, sep="\t", names=["gene", "expression"]) for path in tqdm(paths)],
                            ignore_index=True)
    print("Normalizing expression levels...")
    complete_df.expression = da.log2(complete_df.expression + 1)
    print("Calculating mean and std...")
    complete_df = complete_df.groupby('gene').agg(['mean', 'std']).reset_index()
    print("Filtering useful genes based in thresholds...")
    start = timeit.default_timer()
    with ProgressBar():
        useful_genes = complete_df.loc[(complete_df[('expression', 'mean')] > mean_thr) &
                                       (complete_df[('expression', 'std')] > std_thr)].gene.compute()
    print("There are " + str(len(useful_genes)) + " useful genes")
    print("Saving list of useful genes...")
    useful_genes.to_pickle("useful_genes"+"_"+str(mean_thr)+"_"+str(std_thr)+".pkl")
    print("Saving loc2gene dict...")
    sorted_useful_genes = useful_genes.sort_values().reset_index()
    loc2gene = {}
    for i in range(len(sorted_useful_genes)):
        loc2gene[i] = sorted_useful_genes.loc[i].gene
    with open("loc2gene"+"_"+str(mean_thr)+"_"+str(std_thr)+".pkl", 'wb') as f:
        pickle.dump(loc2gene, f)
    end = timeit.default_timer()
    print("Elapsed time: " + str(end - start))

"""
Code in charge of creating matrices associated to gene expression vectors, labels and ID for each patient, separated
by every type of cancer.
"""

if mode == "threshold" or mode == "go" or mode == "create_matrices":
    matrices_path = "matrices"+"_"+str(mean_thr)+"_"+str(std_thr)
    try:
        os.makedirs(matrices_path)
        print("Matrices directory created")
    except:
        print("Matrices directory already exist")

    useful_genes = pd.read_pickle("useful_genes"+"_"+str(mean_thr)+"_"+str(std_thr)+".pkl")
    for c in class2anot_dict.keys():
        x_matrix, y_vec, id_vec = [], [], []
        txt_paths = glob.glob("data/"+c+"/*/*.txt")
        print("Creating " + c + " matrices")
        for path in tqdm(txt_paths):
            x = pd.read_csv(path, sep="\t", names=["gene", "expression"])
            x = x[x["gene"].isin(useful_genes)].sort_values("gene").reset_index().expression.map(lambda exp: np.log2(exp+1))
            x_matrix.append(x.to_list())
            strings = path.split("/")
            if strings[2] == "Solid Tissue Normal":
                y_vec.append(0)
            else:
                y_vec.append(class2anot_dict[strings[1]])
            id_vec.append(strings[3][:-4])

        try:
            os.makedirs(os.path.join(matrices_path, c))
        except:
            print(matrices_path+"/"+c+" directory already exist")

        with open(os.path.join(matrices_path, c, "x.npy"), 'wb') as f:
            np.save(f, np.array(x_matrix))
        with open(os.path.join(matrices_path, c, "y.npy"), 'wb') as f:
            np.save(f, np.array(y_vec))
        with open(os.path.join(matrices_path, c, "id.npy"), 'wb') as f:
            np.save(f, np.array(id_vec))

def matrixloader(mean_thr, std_thr, min_max_norm=True, shuffle=True):
    """
        This function returns a matrix with (m, n) dimensions where m is the number of samples considered in the
        TCGA dataset and n is the number of useful genes based in the mean and standard deviation thresholds previously
        defined. It also returns a vector with the cancer type annotation for each row, a vector with the ID of the
        sample, and a dictionary that matches every gene vector location to the name of the gene whose expression is
        represented.

        IMPORTANT: The "useful_genes_(mean_thr)_(std_thr).pkl", "loc2gene_(mean_thr)_(std_thr).pkl" and
        "matrices_(mean_thr)_(std_thr)" files and directory should already exist in your current working directory.
        This files can be generated by running the dataloader.py file in "threshold" or "go" mode using the desired
        thresholds as parameters.


        :param mean_thr: Mean threshold for choosing useful genes
        :param std_thr: Standard deviation threshold for choosing useful genes
        :param min_max_norm: Indicates if the returned gene expression matrix will be normalized through min-max method
        by each gene in order to guarantee expression levels between 0 and 1.
        :param shuffle: Indicates if the returned matrix and vectors will be shuffled.

        :return: X: Gene expression matrix of m, n) dimensions where m is the number of samples considered in the
        TCGA dataset and n is the number of useful genes based in the mean and standard deviation thresholds previously
        defined.
        :return: Y: Labels associated to each gene expression matrix row
        :return: ID: Sample ID associated to each gene expression matrix row
        :return: loc2gene: Dictionary that maps gene vector location to the name of the gene whose expression is being
        represented.
    """

    matrices_path = "matrices" + "_" + str(mean_thr) + "_" + str(std_thr)
    loc2gene_path = "loc2gene"+"_"+str(mean_thr)+"_"+str(std_thr)+".pkl"

    with open(loc2gene_path, 'rb') as f:
        loc2gene = pickle.load(f)

    x_paths = glob.glob(matrices_path + "/*/x.npy")
    y_paths = glob.glob(matrices_path + "/*/y.npy")
    id_paths = glob.glob(matrices_path + "/*/id.npy")
    X = np.concatenate(tuple([np.load(path) for path in x_paths]), axis=0)
    Y = np.concatenate(tuple([np.load(path) for path in y_paths]), axis=0)
    ID = np.concatenate(tuple([np.load(path) for path in id_paths]), axis=0)

    if min_max_norm:
        X_norm = np.zeros(X.shape)
        for i in range(X.shape[1]):
            X_norm[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))
        X = X_norm

    if shuffle:
        np.random.seed(1)
        shuffler = np.random.permutation(len(X))
        X = X[shuffler]
        Y = Y[shuffler]
        ID = ID[shuffler]

    return X, Y, ID, loc2gene

def generate_graph_adjacency(X, corr_thr, p_thr=0.05):
    """
        This function defines the adjacency of a graph whose nodes represent each of the considered genes. This
        adjacency is based in the Spearman Correlation. If the absolute value of the correlation between 2 genes is
        greater than corr_thr and its associated p-value is less than p_thr, an edge between the nodes of the 2 genes
        is established with a weight of 1.

        :param X: Gene expression matrix
        :param corr_thr: Correlation threshold
        :param p_thr: P value threshold

        :return: edge_indices: 2D numpy array that defines the edges between nodes of the graph
        :return: edge_attributes: Weights associated to the previously defined edges.
    """
    correlation, p_value = spearmanr(X)
    correlation = np.abs(correlation) > corr_thr
    p_value = p_value < p_thr
    adjacency_matrix = np.logical_and(correlation, p_value).astype(int)
    adjacency_matrix_sparse = coo_matrix(adjacency_matrix)
    adjacency_matrix_sparse.eliminate_zeros()
    edge_indices, edge_attributes = from_scipy_sparse_matrix(adjacency_matrix_sparse)
    print("Connected nodes: " + str(np.sum(np.logical_and(correlation, p_value).any(axis=0))))
    print("Total amount of edges: " + str(len(edge_attributes)))
    return edge_indices, edge_attributes

# Example code for data loading in main
# from dataloader import dataloader, generate_graph_adjacency
# X, Y, ID, loc2gene = dataloader(0.5, 0.8)
# edge_indices, edge_attributes = generate_graph_adjacency(X, 0.6, 0.05)






