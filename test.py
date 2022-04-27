import tqdm
import pandas as pd
import numpy as np
import os
import time
from utils import *



class ToilDataset():
    def __init__(self, path, tcga = True, gtex = True, mean_thr=0.5, std_thr=0.5, corr_thr=0.6, p_thr=0.05,
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
        self.corr_thr = corr_thr
        self.p_thr = p_thr
        self.label_type = label_type # can be 'phenotype' or 'category'
        self.force_compute = force_compute

        # Read data from the Toil data set
        self.matrix_data, self.categories, self.phenotypes = self.read_toil_data()
        # Filter toil datasets to use GTEx, TCGA or both
        self.matrix_data_filtered, self.categories_filtered, self.phenotypes_filtered = self.filter_toil_datasets()
        # Filter genes based on mean and std
        self.filtered_gene_list, self.filtering_info = self.filter_genes()
        # Get labels and label dictionary
        self.label_df, self.category2label = self.find_labels()
    
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
        matrix_data_filtered = self.matrix_data.iloc[:, ~self.matrix_data.columns.str.contains("TARGET")]

        # Handle the filters for TCGA and GTEx
        if self.tcga and ( not self.gtex):
            print("Using TCGA samples only")
            # Filter out all gtex samples from matrix_data
            matrix_data_filtered = matrix_data_filtered.iloc[:, ~matrix_data_filtered.columns.str.contains("GTEX")]
            categories_filtered = self.categories[~self.categories["sample"].str.contains("GTEX")]
            phenotypes_filtered = self.phenotypes[~self.phenotypes["sample"].str.contains("GTEX")]
        elif ( not self.tcga) and self.gtex:
            print("Using GTEX samples only")
            # Filter out all tcga samples from matrix_data
            matrix_data_filtered = matrix_data_filtered.iloc[:, ~matrix_data_filtered.columns.str.contains("TCGA")]
            categories_filtered = self.categories[~self.categories["sample"].str.contains("TCGA")]
            phenotypes_filtered = self.phenotypes[~self.phenotypes["sample"].str.contains("TCGA")]
        elif self.tcga and self.gtex:
            # Do nothing because both TCGA and GTEX samples are included
            print("Using TCGA and GTEX samples")
            matrix_data_filtered = self.matrix_data
            categories_filtered = self.categories
            phenotypes_filtered = self.phenotypes
        else:
            raise ValueError("You are not selecting any dataset.")
            
        return matrix_data_filtered, categories_filtered, phenotypes_filtered
        
    # This function computes the mean and standard deviation of the matrix_data and filters out the samples with mean and standard deviation below the thresholds
    def filter_genes(self):
        if (not os.path.exists(self.dataset_info_path)) or self.force_compute:
            print("Computing mean, std and list of filtered genes. And saving filtering info to:\n"+ os.path.join(self.dataset_info_path, "filtering_info.csv"))
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
            print("Loading filtering info from:\n" + os.path.join(self.dataset_info_path, "filtering_info.csv"))
            filtering_info_df = pd.read_csv(os.path.join(self.dataset_info_path, "filtering_info.csv"), index_col = 0)
            # get indices of filtering_info_df that are True in the included column
            gene_list = filtering_info_df.index[filtering_info_df["included"].values == True]
            # Plot histograms with plot_filtering_histograms()
            self.plot_filtering_histograms(filtering_info_df)

        return gene_list.to_list(), filtering_info_df

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
        category2label = {}
        label_df = 0

        if self.label_type == 'category':
            label_df = self.categories_filtered
            unique_classes = np.sort(label_df.TCGA_GTEX_main_category.unique())
            print(unique_classes)
        elif self.label_type == 'phenotype':
            label_df = self.phenotypes_filtered

        return label_df, category2label

category_mapper = {'GTEX Adipose Tissue':                       'GTEX-ADI',
                   'GTEX Adrenal Gland':                        'GTEX-ADR_GLA',
                   'GTEX Bladder':                              'GTEX-BLA',
                   'GTEX Blood':                                'GTEX-BLO',
                   'GTEX Blood Vessel':                         'GTEX-BLO_VSL',
                   'GTEX Brain':                                'GTEX-BRA',
                   'GTEX Breast':                               'GTEX-BRE',
                   'GTEX Cervix Uteri':                         'GTEX-CER',
                   'GTEX Colon':                                'GTEX-COL',
                   'GTEX Esophagus':                            'GTEX-ESO',
                   'GTEX Fallopian Tube':                       'GTEX-FAL_TUB',
                   'GTEX Heart':                                'GTEX-HEA',
                   'GTEX Kidney':                               'GTEX-KID',
                   'GTEX Liver':                                'GTEX-LIV',
                   'GTEX Lung':                                 'GTEX-LUN',
                   'GTEX Muscle':                               'GTEX-MUS',
                   'GTEX Nerve':                                'GTEX-NER',
                   'GTEX Ovary':                                'GTEX-OVA',
                   'GTEX Pancreas':                             'GTEX-PAN',
                   'GTEX Pituitary':                            'GTEX-PIT',
                   'GTEX Prostate':                             'GTEX-PRO',
                   'GTEX Salivary Gland':                       'GTEX-SAL_GLA',
                   'GTEX Skin':                                 'GTEX-SKI',
                   'GTEX Small Intestine':                      'GTEX-SMA_INT',
                   'GTEX Spleen':                               'GTEX-SPL',
                   'GTEX Stomach':                              'GTEX-STO',
                   'GTEX Testis':                               'GTEX-TES',
                   'GTEX Thyroid':                              'GTEX-THY',
                   'GTEX Uterus':                               'GTEX-UTE',
                   'GTEX Vagina':                               'GTEX-VAG',
                   'TCGA Acute Myeloid Leukemia':               'TCGA-LAML',
                   'TCGA Adrenocortical Cancer':                'TCGA-ACC',
                   'TCGA Bladder Urothelial Carcinoma':         'TCGA-BLCA',
                   'TCGA Brain Lower Grade Glioma':             'TCGA-LGG',
                   'TCGA Breast Invasive Carcinoma':            'TCGA-BRCA',
                   'TCGA Cervical & Endocervical Cancer':       'TCGA-CESC',
                   'TCGA Cholangiocarcinoma':                   'TCGA-CHOL',
                   'TCGA Colon Adenocarcinoma':                 'TCGA-COAD',
                   'TCGA Diffuse Large B-Cell Lymphoma':        'TCGA-DLBC',
                   'TCGA Esophageal Carcinoma':                 'TCGA-ESCA',
                   'TCGA Glioblastoma Multiforme':              'TCGA-GBM',
                   'TCGA Head & Neck Squamous Cell Carcinoma':  'TCGA-HNSC',
                   'TCGA Kidney Chromophobe':                   'TCGA-KICH',
                   'TCGA Kidney Clear Cell Carcinoma':          'TCGA-KIRC',
                   'TCGA Kidney Papillary Cell Carcinoma':      'TCGA-KIRP',
                   'TCGA Liver Hepatocellular Carcinoma':       'TCGA-LIHC',
                   'TCGA Lung Adenocarcinoma':                  'TCGA-LUAD',
                   'TCGA Lung Squamous Cell Carcinoma':         'TCGA-LUSC',
                   'TCGA Mesothelioma':                         'TCGA-MESO',
                   'TCGA Ovarian Serous Cystadenocarcinoma':    'TCGA-OV',
                   'TCGA Pancreatic Adenocarcinoma':            'TCGA-PAAD',
                   'TCGA Pheochromocytoma & Paraganglioma':     'TCGA-PCPG',
                   'TCGA Prostate Adenocarcinoma':              'TCGA-PRAD',
                   'TCGA Rectum Adenocarcinoma':                'TCGA-READ',
                   'TCGA Sarcoma':                              'TCGA-SARC',
                   'TCGA Skin Cutaneous Melanoma':              'TCGA-SKCM',
                   'TCGA Stomach Adenocarcinoma':               'TCGA-STAD',
                   'TCGA Testicular Germ Cell Tumor':           'TCGA-TGCT',
                   'TCGA Thymoma':                              'TCGA-THYM',
                   'TCGA Thyroid Carcinoma':                    'TCGA-THCA',
                   'TCGA Uterine Carcinosarcoma':               'TCGA-UCS',
                   'TCGA Uterine Corpus Endometrioid Carcinoma':'TCGA-UCEC',
                   'TCGA Uveal Melanoma':                       'TCGA-UVM',
                   }

test_toil_dataset = ToilDataset(os.path.join("data", "toil_data"),
                                tcga = True,
                                gtex = True,
                                mean_thr = 0.5,
                                std_thr = 0.5,
                                corr_thr = 0.6,
                                p_thr = 0.05,
                                label_type = 'category',
                                force_compute = False)


