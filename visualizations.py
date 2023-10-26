import os
import pylab
import time
import scanpy as sc
from matplotlib.pyplot import rc_context
# Import auxiliary functions
from utils import *
from datasets import *
from batch_metrics import *


# Set axis bellow for matplotlib
plt.rcParams['axes.axisbelow'] = True
# Set figure fontsizes
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

# Get Parser
parser = get_general_parser()
# Parse the argument
args = parser.parse_args()
args_dict = vars(args)


# Obtain dataset depending on the args specified source
dataset = get_dataset_from_args(args)

# Get adata from current dataset 
adata = get_adata_from_dataset(dataset)

sc.pp.pca(adata, n_comps=50, copy=False)
sc.tl.tsne(adata, n_jobs=-1, copy=False)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X', copy=False)
sc.tl.umap(adata, copy=False)

# Add string is_tcga column
adata.obs['Source'] = adata.obs['is_tcga'].map({False: 'GTEx', True: 'TCGA'})

# Find real processing level to save the image
real_level = args.wang_level
real_level = 4 if real_level == 3 and args.batch_norm == 'std' else real_level
real_level = 5 if real_level == 3 and args.batch_norm == 'mean' else real_level
real_level = 6 if real_level == 3 and args.batch_norm == 'both' else real_level


with rc_context({'figure.figsize': (8, 8)}):
    sc.pl.umap(adata, color=['Source', 'tissue_txt'], save=f'_{args.source}_lev_{real_level}.png', ncols=1, frameon=False)

with rc_context({'figure.figsize': (8, 8)}):
    sc.pl.pca(adata, color=['Source', 'tissue_txt'], save=f'_{args.source}_lev_{real_level}.png', ncols=1, frameon=False)

with rc_context({'figure.figsize': (8, 8)}):
    sc.pl.tsne(adata, color=['Source', 'tissue_txt'], save=f'_{args.source}_lev_{real_level}.png', ncols=1, frameon=False)

