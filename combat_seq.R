library("feather")
library("sva")
library("DESeq2")
library("pracma")
library("ggplot2")
library("matrixStats")
library("edgeR")


# Read data
cat('Starting to read data...\n')
start_time <- Sys.time()
count_matrix <- read_feather(file.path("data", "toil_data", "count_matrix.feather"))
end_time <- Sys.time()
sprintf("Took %0.2f seconds to read data.", end_time-start_time)
count_matrix <- data.frame(count_matrix, row.names = count_matrix$sample)

################################################
# Temporal code to test gene number scalability
# count_matrix <- count_matrix[1:n,]
################################################


count_matrix$sample <- NULL

int_count_matrix <- ceil((2^count_matrix)-1)

gene_means <- rowMeans(int_count_matrix)
gene_std <- rowSds(as.matrix(int_count_matrix))

# int_count_matrix_filtered <- int_count_matrix[gene_means>=100000.0, ]
# row_sub <- apply(int_count_matrix, 1, function(row) all(row !=0 ))
# int_count_matrix_filtered <- int_count_matrix[gene_std>=10000.0, ]

int_count_matrix_filtered <- int_count_matrix[gene_means>0.0, ]

sprintf("There are %0.0f remaining genes after filtering.", dim(int_count_matrix_filtered)[1])

metadata_file <- read.csv(file = file.path("data", "toil_data", "batch_assignment.csv"))

metadata_samples <- metadata_file[,'sample']
cont_matrix_samples <- colnames(count_matrix)
common_samples <- intersect(metadata_samples, gsub("\\.", "-", cont_matrix_samples))

batch_affected_count_matrix <- int_count_matrix_filtered[gsub("-", ".", common_samples)]
batch_affected_count_matrix[] <- lapply(batch_affected_count_matrix, as.integer)

batch_affected_metadata <- subset(metadata_file, sample %in% common_samples)
batch_affected_metadata$batch <- batch_affected_metadata[,'X_study'] == 'TCGA'

# Test DGEList to check if any sample generates library problems
test_dge <-  DGEList(counts = as.matrix(batch_affected_count_matrix))
# Get invalid samples
invalid_samples <- rownames(test_dge$sample[test_dge$samples['lib.size']==0,])
sprintf("There were %0.0f invalid samples.", dim(invalid_samples)[1])


# Remove invalid samples from count_matrix and metadata
valid_batch_affected_count_matrix <- batch_affected_count_matrix[, !names(batch_affected_count_matrix) %in% invalid_samples]
valid_batch_affected_metadata <- subset(batch_affected_metadata, !(sample %in% gsub("\\.", "-", invalid_samples))) 

start_time <- Sys.time()
combat_seq_counts <- ComBat_seq(counts=as.matrix(valid_batch_affected_count_matrix),
                                batch=as.integer(valid_batch_affected_metadata[,'batch']),
                                group=as.integer(valid_batch_affected_metadata[,'lab_num']),
                                full_mod=TRUE)
end_time <- Sys.time()
sprintf("There ComBat-Seq process took %0.0f seconds.", end_time-start_time)


png(file="gene_means.png", width=600, height=350)
ggplot(data.frame(gene_means), aes(x=gene_means)) + geom_histogram(bins=100) + scale_x_log10()
dev.off()

png(file="gene_std.png", width=600, height=350)
ggplot(data.frame(gene_std), aes(x=gene_std)) + geom_histogram(bins=100) + scale_x_log10()
dev.off()