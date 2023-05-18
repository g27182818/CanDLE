library(org.Hs.eg.db)

# Get Entrez IDs to symbol mapping dataframe
eg2sym <- toTable(org.Hs.egSYMBOL)

# Get Entrez IDs to gene Ensembl mapping dataframe
eg2ens <- toTable(org.Hs.egENSEMBL)

# Get the intersection of the two dataframes in the Entrez ID column
glob_mapping <- merge(eg2ens, eg2sym, by="gene_id")

# Sort the gene_id column numerically
glob_mapping <- glob_mapping[order(as.numeric(glob_mapping$gene_id)),]

# Change the name of the gene_id column to Entrez ID
colnames(glob_mapping)[1] <- "entrez_id"

# Overwrite the dataframe to a csv file in the data folder
write.csv(glob_mapping, file=file.path("data", "gene_names.csv"), row.names=FALSE)


