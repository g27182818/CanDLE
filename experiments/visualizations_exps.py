import json
import subprocess
import os

"""
This code visualizes three different dimensionality reduction algorithms (PCA, TSNE, UMAP) in all datasets (wang, toil and recount3) and all the
processing levels (0, 1, 2, 3, 4, 5, 6). The description of the processing is as follows:

Level 0: No processing. The data is just filtered for genes and samples with enough counts.
Level 1: Samples from categories of the GTEx of TCGA that do not have sufficient paired samples in the other dataset are removed.
Level 2: Reverse log2 transform, quantile normalize the data matrix and apply log2(x+1) transform.
Level 3: ComBat batch correction is applied to each tissue type separately. In other words, for a given tissue type (e.g. Lung) we 
         perform ComBat batch correction on the samples of that tissue type from both datasets ('GTEX-LUN', 'TCGA-LUAD', 'TCGA-LUSC').
         The batch variable is the dataset (GTEx or TCGA) and the deceased state (healthy or tumor) is the biologically relevant
         variable (design matrix).
Level 4: Take level 3 and reduce standard deviation of each gene to 1 for each batch. So we compute the standard deviation of each
         gene in TCGA and each gene in GTEx and then we divide each gene by the standard deviation of that gene in its respective
         dataset. So std if 1 for each gene in TCGA and std is 1 for each gene in GTEx.
Level 5: We do the same as in level 4 but instead of the std we zero the mean of each gene in each batch. This effectively centers
         both batch distributions around 0.
Level 6: We do the same as in level 4 but we do both the std and mean correction. This is the standard Z-score normalization. Applied
         to each batch separately.
"""

datasets = ['wang', 'toil', 'recount3']
processing_levels = [1, 2, 3, 4, 5, 6] # NOTE: We are omitting level 0 because we take for granted sample filtering

# Iterate over datasets
for dataset in datasets:
    
    # Iterate over processing levels
    for lev in processing_levels:

        ### Get all the relevant config files. NOTE: In this case, the dataset config is the only relevant config file
        # Read dataset config
        with open(os.path.join('configs', 'datasets', f'config_{dataset}.json'), 'r') as f:
            dataset_config = json.load(f)
        
        # Unify config params
        config_params = dataset_config

        # Modify config params to the specified processing level
        config_params['wang_level'] = lev
        config_params['batch_norm'] = 'None'

        # Handle the level 4
        if lev == 4:
            config_params['wang_level'] = 3
            config_params['batch_norm'] = 'std'
        
        # Handle the level 5
        if lev == 5:
            config_params['wang_level'] = 3
            config_params['batch_norm'] = 'mean'
        
        # Handle the level 6
        if lev == 6:
            config_params['wang_level'] = 3
            config_params['batch_norm'] = 'both'

        # Modify experiment name
        config_params['exp_name'] = 'None'

        # Start building the command
        command_list = ['python', 'visualizations.py']

        # Add all the config params
        for key, val in config_params.items():
            command_list.append(f'--{key}')
            command_list.append(f'{val}')

        print(f'Performind dimensionality reduction for {dataset} at processing level {lev}...')

        # Call subprocess
        subprocess.call(command_list)