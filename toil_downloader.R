library("UCSCXenaTools")
library("feather")

# Print Masage to the user
cat("Starting download of the data\nThis should take approximately 10 minutes...\n\n\n\n")
# Make pause for 10 seconds to let the user read the message
Sys.sleep(10)

# This sets an environment variable to let R load the huge dataset
Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 4)

# Define the data hub, cohort and specific datasets
xe <- XenaHub(hostName = "toilHub",
             cohorts = "TCGA TARGET GTEx",
             datasets = c("TcgaTargetGTEX_phenotype.txt",
                          "TCGA_GTEX_category.txt",
                          "TcgaTargetGtex_rsem_gene_tpm",
                          "TcgaTargetGtex_gene_expected_count"))

# Define the Xena query object
xe_query <- XenaQuery(xe)


# Download data
cat("Downloading data...\n")
xe_download <- XenaDownload(xe_query, destdir = file.path("data", "toil_download"), max_try = 10L)
# Load data into R
cat("Loading data into R...\n")
dat <- XenaPrepare(xe_download, chunk_size = 100)

data_matrix <- dat$TcgaTargetGtex_rsem_gene_tpm.gz
count_matrix <- dat$TcgaTargetGtex_gene_expected_count.gz
categories <- dat$TCGA_GTEX_category.txt
phenotypes <- dat$TcgaTargetGTEX_phenotype.txt.gz

# create data directory
dir.create(file.path("data", "toil_data"), recursive = TRUE, showWarnings = FALSE)

# Save data
cat("Writing data to", file.path("data", "toil_data"), "directory\n")
write_feather(data_matrix, file.path("data", "toil_data", "data_matrix.feather"))
write_feather(count_matrix, file.path("data", "toil_data", "count_matrix.feather"))
write.csv(categories, file.path("data", "toil_data", "categories.csv"))
write.csv(phenotypes, file.path("data", "toil_data", "phenotypes.csv"))

# Delete the toil_download directory
# unlink(file.path("data", "toil_download"), recursive = TRUE, force = TRUE)
