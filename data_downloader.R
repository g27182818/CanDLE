library("feather")
library("TCGAbiolinks")
library("EDASeq")
################################################################################
################## Download and obtain matrices for TCGA #######################
################################################################################
# Get all projects from GDC
gdc_project_list <- getGDCprojects()$id

# Filter just the ones containing "TCGA"
list_tcga <- grep("TCGA", gdc_project_list, value = TRUE)

# Code just for tests
# list_tcga <- c("TCGA-ACC", "TCGA-CHOL")

# Iterate over the list of projects of the TCGA
for (project_name in list_tcga) {
  # Defines the query to the GDC
  query <- GDCquery(project = project_name,
                    data.category = "Transcriptome Profiling",
                    data.type = "Gene Expression Quantification",
                    experimental.strategy = "RNA-Seq",
                    workflow.type = "STAR - Counts")
  # Get metadata matrix
  metadata <- query[[1]][[1]]
  # Download data using api
  GDCdownload(query, method = "api")
  # Downloads data from GDC
  data <- GDCprepare(query,
                     summarizedExperiment = TRUE)
  # Get tpm, counts, patient names, and gene names from summarized experiment object
  tpm_data <- data@assays@data@listData[["tpm_unstrand"]]
  count_data <- data@assays@data@listData[["unstranded"]]
  patient_identifier <- data@colData@rownames
  gene_info_global <- data@rowRanges@elementMetadata@listData
  ens_genes <- gene_info_global[c(5, 6, 7, 9, 10)]
  # Set name of genes column
  gene_id <- ens_genes[["gene_id"]]
  # Add row and column names to tpm and counts matrices
  cols_names <- append("gene_id", patient_identifier, 1)
  tpm_data <- data.frame(cbind(gene_id, tpm_data))
  colnames(tpm_data) <- cols_names
  count_data <- data.frame(cbind(gene_id, count_data))
  colnames(count_data) <- cols_names
  # Create directory to save matrices and metadata
  dir.create(file.path("data", "tcga", project_name), recursive = TRUE, showWarnings = FALSE)
  # Write tpm and count matrices, sample identifiers, gene info and metadata file with feather or csv
  write_feather(tpm_data, file.path("data", "tcga", project_name, "tpm_matrix.feather"))
  write_feather(count_data, file.path("data", "tcga", project_name, "count_matrix.feather"))
  write_feather(metadata, file.path("data", "tcga", project_name, "metadata.feather"))
  write.table(patient_identifier, file.path("data", "tcga", project_name, "patient_identifier.csv"), sep = ",")
  write.table(ens_genes, file.path("data", "tcga", project_name, "ens_genes.csv"), sep = ",")
}

# test_read <- read_feather(file.path("data",'tcga', project_name, "count_matrix.feather"))

# Delete the temporal folder of data and manifest file
print("Deleting temporal data folders and manifest file...")
unlink("GDCdata", recursive = TRUE)
unlink("MANIFEST.txt")
