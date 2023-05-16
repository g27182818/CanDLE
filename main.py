# Import of needed packages
import numpy as np
import os
import torch
import pickle
import argparse
# Import auxiliary functions
from utils import *
from model import *
from datasets import *
# Set matplotlib option to plot while in screen
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set axis bellow for matplotlib
plt.rcParams['axes.axisbelow'] = True



# Get Parser
parser = get_general_parser()
# Parse the argument
args = parser.parse_args()



# Miscellaneous parameters --------------------------------------------------------------------------------------------#
torch.manual_seed(12345)            # Set torch manual seed                                                            #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")       # Set cuda device                          #
# ---------------------------------------------------------------------------------------------------------------------#

# Handle the possibility of an all vs one binary problem
complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']
if args.all_vs_one=='False':
    binary_dict = {}
else:
    binary_dict = {label: 0 for label in complete_label_list}
    binary_dict[args.all_vs_one] = 1

# Declare dataset depending on the source
if args.source == 'toil':
    dataset = ToilDataset(  os.path.join('data', 'toil_data'),          dataset = args.dataset,
                            tissue = args.tissue,                       binary_dict=binary_dict,
                            mean_thr = args.mean_thr,                   std_thr = args.std_thr,
                            rand_frac = args.rand_frac,                 sample_frac=args.sample_frac,
                            gene_list_csv = args.gene_list_csv,         batch_normalization=args.batch_norm,
                            fold_number = args.fold_number,             partition_seed = args.seed,
                            force_compute = False)

elif args.source == 'wang':
    dataset = WangDataset(  os.path.join('data', 'wang_data'),          dataset = args.dataset,
                            tissue = args.tissue,                       binary_dict=binary_dict,
                            mean_thr = args.mean_thr,                   std_thr = args.std_thr,
                            rand_frac = args.rand_frac,                 sample_frac=args.sample_frac,
                            gene_list_csv = args.gene_list_csv,         batch_normalization=args.batch_norm,
                            fold_number = args.fold_number,             partition_seed = args.seed,
                            force_compute = False)

elif args.source == 'recount3':
    dataset = Recount3Dataset(os.path.join('data', 'recount3_data'),    dataset = args.dataset,
                            tissue = args.tissue,                       binary_dict=binary_dict,
                            mean_thr = args.mean_thr,                   std_thr = args.std_thr,
                            rand_frac = args.rand_frac,                 sample_frac=args.sample_frac,
                            gene_list_csv = args.gene_list_csv,         batch_normalization=args.batch_norm,
                            fold_number = args.fold_number,             partition_seed = args.seed,
                            force_compute = False)

# Calculate loss function weights
distribution = np.bincount(np.ravel(dataset.label_df['lab_num']).astype(np.int64))
loss_weights = 2500000 / (distribution**2)
lw_tensor = torch.tensor(loss_weights, dtype=torch.float).to(device)

# Declare all the saving paths needed
path_dict = get_paths(args.exp_name)

# Create results directory
if not os.path.isdir(path_dict['results']):
    os.makedirs(path_dict['results'])

fold_performance = {}

if (args.mode == "train") or (args.mode == 'both'):
    
    for i in range(args.fold_number):

        # Model definition
        model = MLP([dataset.num_genes], out_size = dataset.num_classes ).to(device)
        # Print to console model definition
        print("The model definition is:")
        print(model)

        # Define optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        criterion = torch.nn.CrossEntropyLoss(weight=lw_tensor) if args.weights else torch.nn.CrossEntropyLoss()

        # Dataloader declaration
        train_loader, test_loader = dataset.get_dataloaders(batch_size = args.batch_size, fold=i)

        # Performance lists declarations
        train_metric_lst = []
        test_metric_lst = []
        loss_list = []

        # Stop auxiliary variables
        stop = False
        epoch = 0
        best_macc = -1
        best_epoch = -1

        # Train/test cycle
        while stop == False:
        # for epoch in range(args.epochs):
            print('-'*89)
            print(f'Fold {i+1}, Epoch {epoch+1} :')
            print('\n')
            # Train one epoch
            loss = train(train_loader, model, device, criterion, optimizer)
            
            # Obtain test metrics for each epoch in all groups
            train_metrics = test(train_loader, model, device, num_classes=dataset.num_classes)
            test_metrics = test(test_loader, model, device, num_classes=dataset.num_classes)

            # Add epoch information to the dictionaries
            train_metrics['epoch'], test_metrics['epoch'] = epoch, epoch

            # Append data to list
            train_metric_lst.append(train_metrics), test_metric_lst.append(test_metrics), loss_list.append(loss.cpu().detach().numpy())
            
            # Get if this epoch has the best model
            if test_metrics['mean_acc'] > best_macc:
                best_epoch = epoch
                best_macc = test_metrics['mean_acc']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, os.path.join(path_dict['results'], f'fold_{i+1}_best_model.pt'))

            # Handle args.epochs = -1 to stop until training convergence
            # FIXME: do this with the loss
            if (args.epochs == -1) and ((epoch - best_epoch) > 30):
                # Save last model and stop training if the test metrics have not improved in the las 30 epochs
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, os.path.join(path_dict['results'], f'fold_{i+1}_checkpoint_epoch_{epoch+1}.pt'))
                stop = True

            # Save checkpoint and stop cycle at last epoch
            elif epoch+1== args.epochs:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}, os.path.join(path_dict['results'], f'fold_{i+1}_checkpoint_epoch_{epoch+1}.pt'))
                stop = True
            
            # Print performance
            print_epoch(train_metrics, test_metrics, loss, epoch, i+1, path_dict['train_log'])
            with open(path_dict['train_log'], 'a') as f:
                print_both(f'The best epoch is {best_epoch+1} with mACC of {round(best_macc, 5)}\n',f)

            # Pass to next epoch
            epoch = epoch + 1

        fold_performance[i] = {'train': train_metric_lst, 'test': test_metric_lst, 'loss': loss_list}
    

    with open(path_dict['metrics'], 'wb') as f:
        pickle.dump(fold_performance, f)

    # Generate training performance plot and save it to train_performance_fig_path
    plot_training(fold_performance, path_dict['train_fig'])

    # Generate confusion matrices plot and save it to conf_matrix_fig_path
    plot_conf_matrix(fold_performance, dataset.lab_txt_2_lab_num, path_dict['conf_matrix_fig'])

    # Add a way to print the generalized performance over folds
    print_final_performance(fold_performance, path_dict['train_log'])

    
    # Plot PR curve if the problem is binary
    if dataset.num_classes == 2:
        plot_pr_curve(fold_performance, path_dict['pr_fig'])
    else:
        # If not, make decision confidence plot with the score probabilities
        plot_confidence_violin(fold_performance, dataset.lab_txt_2_lab_num, path_dict['violin_conf_fig'])          
    

# TODO: Solve tha interpretation protocol now that we have k-folds
# TODO: Handle the loading of the best model or the last model. Right now it only tests if there is a fixed number of epochs                        
if ((args.mode == 'test') or (args.mode == 'both')) & (args.epochs>-1):
    # Declare path to load final model
    final_model_path = os.path.join(path_dict['results'], f'checkpoint_epoch_{args.epochs}.pt')

    # Load final model dicts
    total_saved_dict = torch.load(final_model_path)
    model_dict = total_saved_dict['model_state_dict']
    optimizer_dict = total_saved_dict['optimizer_state_dict']

    # Load state dicts to model and optimizer
    model.load_state_dict(model_dict)
    optimizer.load_state_dict(optimizer_dict)

    # Put model in eval mode
    model.eval()

    # Obtain test metrics
    test_metrics = test(test_loader, model, device, num_classes=dataset.num_classes)
    # Print test metrics
    print(f'Test metrics: mean_acc {test_metrics["mean_acc"]} | tot_acc {test_metrics["tot_acc"]} | mean_AP {test_metrics["mean_AP"]}')
    with open(path_dict['train_log'], 'a') as f:
        print(f'Test metrics: mean_acc {test_metrics["mean_acc"]} | tot_acc {test_metrics["tot_acc"]} | mean_AP {test_metrics["mean_AP"]}', file=f)

    # FIXME: Do interpretation only if dataset=='both'
    # This is the code for the interpretation of one model of candle
    # Get model weights 
    weight_matrix = model.out.weight.detach().cpu().numpy()
    # Get indexes where there is a cancer class
    tcga_indexes = [val for key, val in dataset.lab_txt_2_lab_num.items() if 'TCGA' in key]
    tcga_weight_matrix = weight_matrix[tcga_indexes, :]

    # Code to sort gene weights for each class
    gene_names = np.array(dataset.filtered_gene_list) # Get original gene names
    rankings = np.argsort(np.abs(tcga_weight_matrix)) # Obtain rankings based on the absolute value of W
    
    # Declare empty matrix to sort TCGA weights 
    sorted_tcga_weight_matrix = np.zeros_like(tcga_weight_matrix)

    # Cycle to assign sorted weights and order rankings from biggest to lower absolute value
    for i in range(len(sorted_tcga_weight_matrix)):
        sorted_vec = tcga_weight_matrix[i, rankings[i][::-1]]
        sorted_tcga_weight_matrix[i] = sorted_vec
        rankings[i] = rankings[i][::-1]
    
    # Number of genes to be selected as important predictors for each cancer class
    k = 1000

    # Get the top-k important genes in each cancer class
    top_k_ranking = rankings[:, :k]
    # Count the number of times each gene was selected in the top-k for any cancer class
    frecuencies = np.bincount(top_k_ranking.flatten())
    
    # Obtain the ranking of frequencies and sort frequency vector
    frec_rank = np.argsort(frecuencies)[::-1]
    gene_frec_sorted = gene_names[frec_rank]
    frecuencies_sorted = frecuencies[frec_rank]

    # Make a datarfame of interpretation results, print it and save it to file
    rank_frec_df = pd.DataFrame({'gene_name': gene_frec_sorted, 'frec': frecuencies_sorted})
    print('The associated gene importance ranking of this CanDLE model is:')
    print(rank_frec_df)

    # Make directory for Rankings if it does not exist
    if not os.path.exists(path_dict['rankings']):
        os.makedirs(path_dict['rankings'])
        
    # Save weights to csv
    pd.DataFrame(rank_frec_df).to_csv(path_dict['1_ranking'])

    # Make scatter plot of weights of a single random class
    rand_int = np.random.randint(0,len(tcga_indexes))
    threshold = sorted_tcga_weight_matrix[rand_int, k]
    plt.figure()
    plt.plot(tcga_weight_matrix[rand_int], '.k', markersize=2, alpha=0.4)
    plt.plot([0, len(tcga_weight_matrix[rand_int])],[threshold, threshold], '--r')
    plt.plot([0, len(tcga_weight_matrix[rand_int])],[-threshold, -threshold], '--r')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(0, len(tcga_weight_matrix[i]))
    plt.xlabel('Gene', fontsize=16)
    plt.ylabel('$w ($Gene$)$', fontsize='large')
    plt.title(f'Weights for class {rand_int}', fontsize='xx-large')
    plt.show()
    plt.tight_layout()

    # Make directory to save random weights plot
    if not os.path.exists(path_dict['figures']):
        os.makedirs(path_dict['figures'])

    plt.savefig(path_dict['weights_demo_fig'], dpi=300)
