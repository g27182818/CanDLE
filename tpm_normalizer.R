library("feather")
library("EDASeq")

# Function to normalize raw count matrix to TPM values
tpm <- function(counts, len) {
  x <- counts / len
  return(t(t(x) * 1e6 / colSums(x)))
}

project_name <- "TCGA-BRCA"
matrix <- read_feather(file.path("data", "tcga", "counts", project_name, "matrix.feather"))
gene_list <- matrix$X1
gene_length <- getGeneLengthAndGCContent(gene_list, org = "hsa", mode = "biomart")
