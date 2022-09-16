import subprocess
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import json
from model import *
from datasets import *
import pickle as pkl
import matplotlib
from adjustText import adjust_text

mode = 'interpret' # 'compute' or 'interpret'
gpu = '0'
num_ranges = np.arange(40, 100)

exp_folder_name = 'bootstrapping'

os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# Get mapper file to know labels
mapper_path = os.path.join('data','toil_data', 'mappers', 'category_mapper.json')
with open(mapper_path, "r") as f:
    category_mapper = json.load(f)

labels = list(category_mapper.values())

# Delete all label that start with 'GTEX'
labels = [label for label in labels if not label.startswith('GTEX')]



# Declare folder names for all experiments
exp_names = [os.path.join(exp_folder_name, 'experiment_'+str(i)) for i in num_ranges]


# If mode is 'compute', compute all experiments
if mode == 'compute':
    for i in range(len(num_ranges)):
        # Just compute the models if they are not already computed
        if not os.path.exists(os.path.join('Results', exp_names[i])):
            # run main.py with subprocess
            command = 'python main.py --exp_name {} --seed {}'.format(exp_names[i], num_ranges[i])
            print(command)
            command = command.split()
            subprocess.call(command)

if mode == 'interpret':

    # Declare dataset
    dataset = ToilDataset(os.path.join("data", "toil_data"),
                        dataset = 'both',
                        tissue = 'all',
                        binary_dict={},
                        mean_thr = -10.0,
                        std_thr = 0.0,
                        use_graph = False,
                        corr_thr = 0.6,
                        p_thr = 0.05,
                        label_type = 'phenotype',
                        batch_normalization='normal',
                        partition_seed = 0,
                        force_compute = False)

    # Get paths from each trained model
    final_model_paths = glob.glob(os.path.join('Results', exp_folder_name, 'experiment_*', "checkpoint_epoch_*.pt"))
    
    matrix_list = []
    # Cycle through paths to get matrices
    for i, path in enumerate(final_model_paths):
        # Load final model dicts
        total_saved_dict = torch.load(path)
        model_dict = total_saved_dict['model_state_dict']
        act_matrix = model_dict['out.weight'].detach().cpu().numpy()
        matrix_list.append(act_matrix)
    
    weight_matrices = np.stack(matrix_list, axis = 2)
    tcga_weight_matrices = weight_matrices[30:, :, :]
    tcga_weight_means = np.mean(tcga_weight_matrices, axis=2, keepdims=True)
    tcga_weight_std = np.std(tcga_weight_matrices, axis=2, keepdims=True)
    z_wald_stat = np.divide(tcga_weight_means, tcga_weight_std, out=np.zeros_like(tcga_weight_means), where=tcga_weight_std!=0)

    # Maximum p-value defined such that (1-max_p)*55602 ~ 95%
    max_p = 1e-6
    # Get indices of weights that pass the Wald z test
    valid_weights = np.squeeze(st.norm.sf(abs(z_wald_stat))*2 < max_p)

    # Zero out gene weights that do not pass the Wald z test
    zeroed_tcga_weight_means = tcga_weight_means 
    zeroed_tcga_weight_means[~valid_weights] = 0
    zeroed_tcga_weight_means = np.squeeze(zeroed_tcga_weight_means)
    
    # Get the gene names
    gene_names = np.array(dataset.filtered_gene_list)

    # Obtain rankings in the absolute values of weights
    rankings = np.argsort(np.abs(zeroed_tcga_weight_means))
    
    # Declare empty sorted weight matrix
    sorted_tcga_weight_matrix = np.zeros_like(zeroed_tcga_weight_means)
    # Empty matrix for the sorted valid salples using Wald z test 
    sorted_valid_matrix = np.zeros_like(zeroed_tcga_weight_means)

    # Cycle over rows of the weight matrix to order each row
    for i in range(len(sorted_tcga_weight_matrix)):
        sorted_vec = zeroed_tcga_weight_means[i, rankings[i][::-1]] # Sort the weights
        sorted_valid = valid_weights[i, rankings[i][::-1]] # Sort the valid samples
        sorted_tcga_weight_matrix[i] = sorted_vec # Assign to the sorted matrix
        sorted_valid_matrix[i] = sorted_valid # Assign to the sorted matrix
        rankings[i] = rankings[i][::-1] # Reverse ranking row
    
    # Get the minimum ranking with an invalid 
    min_invalid_ranking = np.where(np.sum(sorted_valid_matrix, axis=0) == 0)[0][0]
    print('There are {} valid weights using Wald z test and the minimum ranking of an invalid weight is {}'.format(valid_weights.sum(), min_invalid_ranking))

    # Define k threshold
    k = 1000

    top_k_ranking = rankings[:, :k]
    frecuencies = np.bincount(top_k_ranking.flatten())

    frec_rank = np.argsort(frecuencies)[::-1]
    gene_frec_sorted = gene_names[frec_rank]
    frecuencies_sorted = frecuencies[frec_rank]

    rank_frec_df = pd.DataFrame({'gene_name': gene_frec_sorted, 'frec': frecuencies_sorted})
    print(rank_frec_df)
    pd.DataFrame(rank_frec_df).to_csv('100_candle_weights.csv')
