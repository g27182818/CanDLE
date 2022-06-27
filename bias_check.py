from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from datasets import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report

dataset_to_check = 'toil'

if dataset_to_check=='toil':
    ################ Temporal parser code #######################
    ################ Must be replace by configs #################
    # Import the library
    import argparse
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--dataset',        type=str,   default="both",         help="Dataset to use",                                                                                                  choices=["both", "tcga", "gtex"])
    parser.add_argument('--tissue',         type=str,   default="all",          help="Tissue to use from data",                                                                                         choices=['all', 'Bladder', 'Blood', 'Brain', 'Breast', 'Cervix', 'Colon', 'Connective', 'Esophagus', 'Kidney', 'Liver', 'Lung', 'Not Paired', 'Ovary', 'Pancreas', 'Prostate', 'Skin', 'Stomach', 'Testis', 'Thyroid', 'Uterus'])
    parser.add_argument('--all_vs_one',     type=str,   default='False',        help="If False solves a multiclass problem, if other string solves a binary problem with this as the positive class.", choices=['False', 'GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM'])

    parser.add_argument('--model',          type=str,   default="MLP_ALL",      help="Model to use. Baseline is a graph neural network",                                                                choices=["MLP_ALL", "MLP_FIL", "BASELINE"])

    parser.add_argument('--lr',             type=float, default=0.00001,        help="Learning rate")
    parser.add_argument('--batch_size',     type=int,   default=100,            help="Batch size")
    parser.add_argument('--epochs',         type=int,   default=20,             help="Number of epochs")
    parser.add_argument('--adv_e_test',     type=float, default=0.01)
    parser.add_argument('--adv_e_train',    type=float, default=0.00)
    parser.add_argument('--n_iters_apgd',   type=int,   default=50)
    parser.add_argument('--mode',           type=str,   default="test")
    parser.add_argument('--num_test',       type=int,   default=69)
    parser.add_argument('--train_samples',  type=int,   default=-1,             help='Number of samples used for training the algorithm. -1 to run with all data.') # TODO: Program subsampling in dataset. In this moment this still does not work
    parser.add_argument('--exp_name',       type=str,   default='misc_test',    help="Experiment name to save")
    # Parse the argument
    args = parser.parse_args()
    #############################################################

    # ------------------- Important variable parameters -------------------------------------------------------------------#
    # Miscellaneous parameters --------------------------------------------------------------------------------------------#
    torch.manual_seed(12345)            # Set torch manual seed                                                            #
    device = torch.device("cuda")       # Set cuda device                                                                  #
    mode = args.mode                    # Mode to run in code submission can be "test" or "demo"                           #
    num_test = args.num_test            # Number of demo data to plot                                                      #
    # Dataset parameters --------------------------------------------------------------------------------------------------#
    val_fraction = 0.2                  # Fraction of the data used for validation                                         #
    train_smaples = args.train_samples  # Number of samples used for training the algorithm. -1 to run with all data.      #
    dataset = args.dataset              # Dataset to use can be "both", "tcga" or "gtex"                                   #
    tissue = args.tissue                # Tissue to use from data. "all" to use all tissues                                #
    batch_size = args.batch_size        # Batch size parameter                                                             #
    coor_thr = 0.6                      # Spearman correlation threshold for declaring graph topology                      #
    p_value_thr = 0.05                  # P-value Spearman correlation threshold for declaring graph topology              #
    all_vs_one = args.all_vs_one        # If False multiclass problem else defines the positive class for binary problem   #
    # Model parameters ----------------------------------------------------------------------------------------------------#
    hidd = 8                            # Hidden channels parameter for baseline model                                     #
    model_type = args.model             # Model type, can be "MLP_FIL", "MLP_ALL", "BASELINE"                              #
    # Training parameters -------------------------------------------------------------------------------------------------#
    experiment_name = args.exp_name     # Experiment name to define path were results are stored                           #
    lr = args.lr                        # Learning rate of the Adam optimizer (was changed from 0.001 to 0.00001)          #
    total_epochs = args.epochs          # Total number of epochs to train                                                  #
    metric = 'both'                     # Evaluation metric for experiment. Can be 'acc', 'mAP' or 'both'                  #
    train_eps = args.adv_e_train        # Adversarial epsilon for train                                                    #
    n_iters_apgd = args.n_iters_apgd    # Number of performed APGD iterations in train                                     #
    # Test parameters -----------------------------------------------------------------------------------------------------#
    test_eps = args.adv_e_test          # Adversarial epsilon for test                                                     #
    # ---------------------------------------------------------------------------------------------------------------------#

    # Handle input filtering depending in model type
    if model_type == "MLP_FIL":
        mean_thr = 3.0  # Mean threshold for filtering input genes
        std_thr = 0.5   # Standard deviation threshold for filtering input genes
        use_graph = False
    elif model_type == "BASELINE":
        mean_thr = 3.0  
        std_thr = 0.5   
        use_graph = True
    elif model_type == "MLP_ALL":
        mean_thr = -10.0  
        std_thr = 0.1    
        use_graph = False
    else:
        raise NotImplementedError

    # Handle the posibility of an all vs one binary problem
    complete_label_list = ['GTEX-ADI', 'GTEX-ADR_GLA', 'GTEX-BLA', 'GTEX-BLO', 'GTEX-BLO_VSL', 'GTEX-BRA', 'GTEX-BRE', 'GTEX-CER', 'GTEX-COL', 'GTEX-ESO', 'GTEX-FAL_TUB', 'GTEX-HEA', 'GTEX-KID', 'GTEX-LIV', 'GTEX-LUN', 'GTEX-MUS', 'GTEX-NER', 'GTEX-OVA', 'GTEX-PAN', 'GTEX-PIT', 'GTEX-PRO', 'GTEX-SAL_GLA', 'GTEX-SKI', 'GTEX-SMA_INT', 'GTEX-SPL', 'GTEX-STO', 'GTEX-TES', 'GTEX-THY', 'GTEX-UTE', 'GTEX-VAG', 'TCGA-ACC', 'TCGA-BLCA', 'TCGA-BRCA', 'TCGA-CESC', 'TCGA-CHOL', 'TCGA-COAD', 'TCGA-DLBC', 'TCGA-ESCA', 'TCGA-GBM', 'TCGA-HNSC', 'TCGA-KICH', 'TCGA-KIRC', 'TCGA-KIRP', 'TCGA-LAML', 'TCGA-LGG', 'TCGA-LIHC', 'TCGA-LUAD', 'TCGA-LUSC', 'TCGA-MESO', 'TCGA-OV', 'TCGA-PAAD', 'TCGA-PCPG', 'TCGA-PRAD', 'TCGA-READ', 'TCGA-SARC', 'TCGA-SKCM', 'TCGA-STAD', 'TCGA-TGCT', 'TCGA-THCA', 'TCGA-THYM', 'TCGA-UCEC', 'TCGA-UCS', 'TCGA-UVM']

    binary_dict = {}
    for label in complete_label_list:
        if label[:4] == 'GTEX':
            binary_dict[label] = 0
        else:
            binary_dict[label] = 1



    # Declare dataset
    dataset = ToilDataset(os.path.join("data", "toil_data"),
                                dataset = dataset,
                                tissue = tissue,
                                binary_dict=binary_dict,
                                mean_thr = mean_thr,
                                std_thr = std_thr,
                                use_graph = use_graph,
                                corr_thr = coor_thr,
                                p_thr = p_value_thr,
                                label_type = 'phenotype',
                                batch_normalization='normal',
                                partition_seed = 0,
                                force_compute = False)



    labels = dataset.split_labels
    data = dataset.split_matrices

    x_train = data['train'].T
    y_train = labels['train'].index.str.contains('TCGA')



    x_val = data['val'].T
    y_val = labels['val'].index.str.contains('TCGA')

    clf = LinearSVC(random_state=0, verbose=4, max_iter=100000)
    clf.fit(x_train, y_train)

    # Get predictions
    y_pred = clf.predict(x_val)

    print(classification_report(y_val, y_pred))

elif dataset_to_check=='wang':

    dataset = WangDataset(os.path.join('data', 'wang_data'))
    train_index, test_index = train_test_split(dataset.categories["is_tcga"], test_size = 0.2, random_state = 0, stratify = dataset.categories["is_tcga"].values)
    complete_dataset = dataset.matrix_data
    del complete_dataset['Hugo_Symbol']
    del complete_dataset['Gene_Hugo_Symbol']
    x_train, x_test = complete_dataset.iloc[2:, train_index.index], complete_dataset.iloc[2:, test_index.index]
    y_train = dataset.categories.loc[train_index.index, ['sample', 'is_tcga']].set_index('sample')
    y_test = dataset.categories.loc[test_index.index, ['sample', 'is_tcga']].set_index('sample')

    x_train = x_train.T.values
    x_test = x_test.T.values

    # Transforms
    x_train = np.log2(x_train+1)
    x_test = np.log2(x_test+1)

    y_train = np.ravel(y_train.values)
    y_test = np.ravel(y_test.values)

    # Random test
    # y_train = np.random.randint(2, size=len(y_train))
    # y_test = np.random.randint(2, size=len(y_test))

    clf = LinearSVC(random_state=0, verbose=4, max_iter=100000)
    clf.fit(x_train, y_train)

    # Get predictions
    y_pred = clf.predict(x_test)

    print(classification_report(y_test, y_pred))

