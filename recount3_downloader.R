library('recount3')
library('recount')
library("feather")
library('pacman')

# If test code just two accessions will be taken
test_code <- FALSE

# Get all the human projects in Recount 3
human_projects <- available_projects()

# Get all gtex and tcga studies
gtex_studies <- subset(human_projects, file_source == 'gtex')
tcga_studies <- subset(human_projects, file_source == 'tcga')

# Get the accessions of each project
gtex_accessions <- gtex_studies$project
tcga_accessions <- tcga_studies$project

# If test mode just work with the first two accessions 
if (test_code){
  gtex_accessions <- c('FALLOPIAN_TUBE', 'BLADDER') # Set just 2 studies with a really low number of samples (9 and 21)
  tcga_accessions <- c('CHOL', 'DLBC') # Set just 2 studies with a really low number of samples (45 and 48)
}

# Set progress bar
pacman::p_load(progress)
pb <- progress_bar$new(total=length(gtex_accessions), width=120, clear=F,
	format = " Downloading GTEx [:bar] :percent ETA= :eta\n")
pb$tick(0)
# Cycle to download all gtex selected datasets 
for (i in 1:length(gtex_accessions)){
  proj_info <- subset(human_projects, project == gtex_accessions[i] & project_type == "data_sources")   # Get project info
  rse_gene <- create_rse(proj_info, type = 'gene', annotation = "gencode_v29")                          # Get range summarized experiment object
  sample_metadata <- colData(rse_gene)[, grepl('gtex', names(colData(rse_gene)), perl = TRUE)]          # Get sample metadata
  assays(rse_gene)$counts <- transform_counts(rse_gene)                                                 # Scale the counts using the AUC
  assays(rse_gene)$TPM <- recount::getTPM(rse_gene)                                                     # Compute TPM normalization
  tpm_data <- assays(rse_gene)$TPM
  # Merge actual data with previous
  if (i == 1){
    gtex_global_metadata <- sample_metadata
    gtex_global_data <- tpm_data
  }
  else{
    gtex_global_metadata <- rbind(gtex_global_metadata, sample_metadata)
    gtex_global_data <- cbind(gtex_global_data, tpm_data)
  }
  # Advance progress bar one step
  pb$tick()
}

pb1 <- progress_bar$new(total=length(tcga_accessions), width=120, clear=F,
	format = " Downloading TCGA [:bar] :percent ETA= :eta\n")
pb1$tick(0)
# Cycle to download all tcga selected datasets 
for (i in 1:length(tcga_accessions)){
  proj_info <- subset(human_projects, project == tcga_accessions[i] & project_type == "data_sources")   # Get project info
  rse_gene <- create_rse(proj_info, type = 'gene', annotation = "gencode_v29")                          # Get range summarized experiment object
  sample_metadata <- colData(rse_gene)[, grepl('tcga', names(colData(rse_gene)), perl = TRUE)]          # Get sample metadata
  assays(rse_gene)$counts <- transform_counts(rse_gene)                                                 # Scale the counts using the AUC
  assays(rse_gene)$TPM <- recount::getTPM(rse_gene)                                                     # Compute TPM normalization
  tpm_data <- assays(rse_gene)$TPM
  # Merge actual data with previous
  if (i == 1){
    tcga_global_metadata <- sample_metadata
    tcga_global_data <- tpm_data
  }
  else{
    tcga_global_metadata <- rbind(tcga_global_metadata, sample_metadata)
    tcga_global_data <- cbind(tcga_global_data, tpm_data)
  }
  # Advance progress bar one step
  pb1$tick()
}

# Merge both gtex and tcga
global_data <- cbind(gtex_global_data, tcga_global_data)

# create data directory
dir.create(file.path("data", "recount3_data"), recursive = TRUE, showWarnings = FALSE)

# Save data
cat("Writing data to", file.path("data", "recount3_data"), "directory\n")
write_feather(as.data.frame(global_data), file.path("data", "recount3_data", "data_matrix.feather"))
write.csv(gtex_global_metadata, file.path("data", "recount3_data", "gtex_metadata.csv"))
write.csv(tcga_global_metadata, file.path("data", "recount3_data", "tcga_metadata.csv"))
write.csv(rowRanges(rse_gene), file.path("data", "recount3_data", "gene_metadata.csv"))
