import json
import subprocess
import argparse
import os

datasets = ['toil', 'recount3'] #, 'wang'] # TODO: implement all levels of processing for wang
processing_levels = [0, 1, 2, 3, 4]

# Iterate over processing levels
for lev in processing_levels:

    # Iterate over datasets
    for dataset in datasets:
        ### Get all the relevant config files
        
        # Read dataset config
        with open(os.path.join('configs', 'datasets', f'config_{dataset}.json'), 'r') as f:
            dataset_config = json.load(f)
        
        # Read model config
        with open(os.path.join('configs', 'models', f'config_bias.json'), 'r') as f:
            model_config = json.load(f)

        # Read training config
        with open(os.path.join('configs', 'models', f'config_bias.json'), 'r') as f:
            training_config = json.load(f)
        
        # Unify config params
        config_params = {**dataset_config, **model_config, **training_config}

        # Modify config params to the specified processing level
        config_params['wang_level'] = lev
        config_params['batch_norm'] = False

        # Handle the last level
        if lev == 4:
            config_params['wang_level'] = 3
            config_params['batch_norm'] = True

        # Modify experiment name
        config_params['exp_name'] = f'{dataset}_level_{lev}'

        # Start building the command
        command_list = ['python', 'bias_check.py']

        # Add all the config params
        for key, val in config_params.items():
            command_list.append(f'--{key}')
            command_list.append(f'{val}')

        print(f'Doing bias check for {dataset} at processing level {lev}...')

        # Call subprocess
        subprocess.call(command_list)