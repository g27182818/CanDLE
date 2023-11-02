import json
import subprocess
import argparse

# Get parsed the path of the config file
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_config', type=str, default='config_dataset.json',    help='Path to the .json file with the configs of the dataset.')
parser.add_argument('--model_config',   type=str, default='config_model.json',      help='Path to the .json file with the configs of the model.')
parser.add_argument('--train_config',   type=str, default='config_train.json',      help='Path to the .json file with the configs of the training.')
args = parser.parse_args()


# Read the dataset, model and train configs
with open(args.dataset_config, 'rb') as f:
    config_params = json.load(f)

with open(args.model_config, 'rb') as f:
    config_params.update(json.load(f))

with open(args.train_config, 'rb') as f:
    config_params.update(json.load(f))


# Create the command to run. If sota key is "None" call main.py else call main_sota.py
if config_params['sota']=='None':
    command_list = ['python', 'main.py']
elif config_params['sota']=='bias':
    command_list = ['python', 'bias_check.py']
elif config_params['sota']=='rf_auto_ml':
    command_list = ['python', 'rf_auto_ml.py']
else:
    command_list = ['python', 'ml_baseline.py']

for key, val in config_params.items():
    command_list.append(f'--{key}')
    command_list.append(f'{val}')

# Call subprocess
subprocess.call(command_list)