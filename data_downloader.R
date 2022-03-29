library("feather")
library("TCGAbiolinks")
library("EDASeq")
################################################################################
################# Obtain Gene lengths mapper using bioMart #####################
################################################################################
# Check if the gene lengths file is already present
if (file.exists(file.path("data", "gene_mapper.feather")) = FALSE) {
  # Define random project name
  project_name <- "TCGA-ACC"
  # Define the GDC query
  query <- GDCquery(project = project_name,
                    data.category = "Transcriptome Profiling",
                    data.type = "Gene Expression Quantification",
                    experimental.strategy = "RNA-Seq",
                    workflow.type = "HTSeq - Counts")
  # Download data using api
  GDCdownload(query, method = "api")
  # Read downloaded data and get a single matrix for the complete project
  data <- GDCprepare(query,
                     summarizedExperiment = TRUE)
  # Get original genes in TCGA project (have version number)
  tcga_original_genes <- data@rowRanges@elementMetadata@listData[["original_ensembl_gene_id"]]
  # Get ensemble names of original genes (without version number)
  tcga_ens_genes <- data@rowRanges@elementMetadata@listData[["ensembl_gene_id"]]
  # Define parameters for biomart lengths query
  chunk_size <- 100
  start_index <- 1
  end_index <- chunk_size
  gene_lengths <- NULL
  # Cycle to get gene lengths of all tcga_ens genes
  while (start_index < length(tcga_ens_genes)){
    # Get a gene names list of size chunk_size
    actual_gene_list <- tcga_ens_genes[start_index:end_index]
    # Get that list lengths with function
    actual_legths <- getGeneLengthAndGCContent(actual_gene_list, org = "hsa", mode = "biomart")
    # Append result to gene_lengths
    gene_lengths <- rbind(gene_lengths, actual_legths)
    # Print progress
    print("Gene length computing progress (%) is:")
    print(round(100 * end_index / length(tcga_ens_genes), digits = 3))
    # Update start and end index
    start_index <- start_index + chunk_size
    end_index <- min(end_index + chunk_size, length(tcga_ens_genes))
  }
  ###### Save gene lengths to feather file ######
  # Declare mapper dataframe
  gene_mapper <- data.frame(tcga_original_genes,
                            tcga_ens_genes,
                            gene_lengths[, 1])
  # Set colnames and rownames in mapper dataframe
  colnames(gene_mapper) <- c("original-tcga", "ensg", "length")
  rownames(gene_mapper) <- NULL
  # Filter out genes with NA length (not found by bioMart)
  gene_mapper <- gene_mapper[complete.cases(gene_mapper), ]
  # Create data directory
  dir.create(file.path("data"), recursive = TRUE, showWarnings = FALSE)
  # write gene mapper dataset to feather file
  write_feather(gene_mapper, file.path("data","gene_mapper.feather"))
}

################################################################################
################## Download and obtain matrices for TCGA #######################
################################################################################
# Get all projects from GDC
gdc_project_list <- getGDCprojects()$id
# Filter just the ones containing "TCGA"
list_tcga <- grep("TCGA", gdc_project_list, value = TRUE)
# Iterate over the list of projects of the TCGA
for (project_name in list_tcga) {
  # Defines the query to the GDC
  query <- GDCquery(project = project_name,
                    data.category = "Transcriptome Profiling",
                    data.type = "Gene Expression Quantification",
                    experimental.strategy = "RNA-Seq",
                    workflow.type = "HTSeq - Counts")
  # Download data using api
  GDCdownload(query, method = "api", files.per.chunk = 200)
  # Read downloaded data and get a single matrix for the complete project
  data <- GDCprepare(query,
                     summarizedExperiment = FALSE)
  # Remove last 5 rows of data that don't have gene information
  data <- head(data, -5)
  # Get metadata matrix
  metadata <- query[[1]][[1]]
  # Create directory to save matrices and metadata
  dir.create(file.path("data", "tcga", "counts", project_name), recursive = TRUE, showWarnings = FALSE)
  # Write data matrix and metadata file
  write_feather(data, file.path("data", "tcga", "counts", project_name, "matrix.feather"))
  write_feather(metadata, file.path("data", "tcga", "counts", project_name, "metadata.feather"))
}

# Delete the temporal folder of data and manifest file
print("Deleting temporal data folders and manifest file...")
unlink("GDCdata", recursive = TRUE)
unlink("MANIFEST.txt")

