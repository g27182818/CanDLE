library("UCSCXenaTools")
library("feather")
# setwd("C:/Users/Usuario")

# This sets an environment variable to let R load the huge dataset
Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 4)

# Define the data hub, cohort and specific datasets
xe <- XenaHub(hostName = "toilHub",
             cohorts = "TCGA TARGET GTEx",
             datasets = c("TcgaTargetGTEX_phenotype.txt",
                          "TCGA_GTEX_category.txt",
                          "TcgaTargetGtex_rsem_gene_tpm"))

# Define the Xena query object
xe_query <- XenaQuery(xe)

# Download data
xe_download <- XenaDownload(xe_query, destdir = file.path("data", "toil"), max_try = 10L)
# Load data into R
dat <- XenaPrepare(xe_download, chunk_size = 100)

data_matrix <- dat$TcgaTargetGtex_rsem_gene_tpm.gz
categories <- dat$TCGA_GTEX_category.txt
phenotypes <- dat$TcgaTargetGTEX_phenotype.txt.gz


# create data directory
dir.create(file.path("data", "final_toil"), recursive = TRUE, showWarnings = FALSE)

write_feather(data_matrix, file.path("data", "final_toil", "data_matrix.feather"))
write.csv(categories, file.path("data", "final_toil", "categories.csv"))
write.csv(phenotypes, file.path("data", "final_toil", "phenotypes.csv"))
