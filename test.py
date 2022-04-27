from unicodedata import category
import pandas as pd
import numpy as np
import os
import time
from utils import *


def read_toil_data(path):
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
    matrix_data = pd.read_feather(os.path.join(path, "data_matrix.feather"))
    categories = pd.read_csv(os.path.join(path, "categories.csv"), encoding = "cp1252")
    phenotypes = pd.read_csv(os.path.join(path, "phenotypes.csv"), encoding = "cp1252")
    # Delete the first column of categories and phenotypes
    categories = categories.drop(categories.columns[0], axis = 1)
    phenotypes = phenotypes.drop(phenotypes.columns[0], axis = 1)
    end = time.time()
    print("Time to load data: {} s".format(round(end - start, 3)))
    return matrix_data, categories, phenotypes


def filter_toil_datasets(matrix_data, categories, phenotypes, tcga = True, gtex = True):
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
    matrix_data_filtered = matrix_data.iloc[:, ~matrix_data.columns.str.contains("TARGET")]

    # Handle the filters for TCGA and GTEx
    if tcga and (~gtex):
        # Filter out all gtex samples from matrix_data
        matrix_data_filtered = matrix_data_filtered.iloc[:, ~matrix_data_filtered.columns.str.contains("GTEX")]
        categories_filtered = categories[~categories["sample"].str.contains("GTEX")]
        phenotypes_filtered = phenotypes[~phenotypes["sample"].str.contains("GTEX")]
    elif (~tcga) and gtex:
        # Filter out all tcga samples from matrix_data
        matrix_data_filtered = matrix_data_filtered.iloc[:, ~matrix_data_filtered.columns.str.contains("TCGA")]
        categories_filtered = categories[~categories["sample"].str.contains("TCGA")]
        phenotypes_filtered = phenotypes[~phenotypes["sample"].str.contains("TCGA")]
    elif tcga and gtex:
        # Do nothing because both TCGA and GTEX samples are included
        pass
    else:
        raise ValueError("You are not selecting any dataset.")
        
    return matrix_data_filtered, categories_filtered, phenotypes_filtered 


test_matrix_data, test_categories, test_phenotypes = read_toil_data(os.path.join("data", "toil_data"))
x_np, y_np, sample2loc, label2cat = filter_toil_datasets(test_matrix_data, test_categories, test_phenotypes)


