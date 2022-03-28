# setwd("C:/Users/Usuario")
library('TCGAbiolinks')
library('feather')

# Get all projects from GDC
GDC_project_list <- getGDCprojects()$id

# Filter just the ones containing "TCGA"
list_tcga <- grep("TCGA", GDC_project_list, value = TRUE)

# Iterate over the list of projects of the TCGA
for (project_name in list_tcga){
  
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
  
  # Get metadata matrix
  metadata <- query[[1]][[1]]
  
  # Create directory to save matrices and metadata
  dir.create(file.path("data",'tcga', project_name), recursive=TRUE, showWarnings = FALSE)
  # Write data matrix and metadata file
  write_feather(data, file.path("data",'tcga', project_name, "matrix.feather"))
  write_feather(metadata, file.path("data",'tcga', project_name, "metadata.feather"))  
}