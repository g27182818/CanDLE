import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sn
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.colors as colors
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab

# Set figure fontsizes
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)



def train(train_loader, model, device, criterion, optimizer):
    """
    This function performs 1 training epoch in a graph classification model with the possibility of adversarial
    training using the attach function.
    :param train_loader: (torch.utils.data.DataLoader) Pytorch dataloader containing training data.
    :param model: (torch.nn.Module) The prediction model.
    :param device: (torch.device) The CUDA or CPU device to parallelize.
    :param criterion: (torch.nn loss function) Loss function to optimize (For this task CrossEntropy is used).
    :param optimizer: (torch.optim.Optimizer) The model optimizer to minimize the loss function (For this task Adam is
                       used)
    :return: mean_loss: (torch.Tensor) The mean value of the loss function over the epoch.
    """
    # Put model in train mode
    model.train()
    # Start the mean loss value
    mean_loss = 0
    # Start a counter
    count = 0
    with tqdm(train_loader, unit="batch") as t_train_loader:
        
        # Training cycle over the complete training batch
        for data in t_train_loader:  # Iterate in batches over the training dataset.
            t_train_loader.set_description(f"Batch {count+1}")
            # Get the inputs of the model (x) and the groundtruth (y)
            input_x, input_y = data
            input_x, input_y = input_x.to(device), input_y.to(device)

            # Perform a single forward pass.
            out = model(input_x) 
            loss = criterion(out, input_y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            mean_loss += loss
            count += 1

            # Update terminal descriptor
            t_train_loader.set_postfix(loss=loss.item())

    mean_loss = mean_loss/count
    return mean_loss


def test(loader, model, device, num_classes=34):
    """
    This function calculates a set of metrics using a model and its inputs.
    :param loader: (torch.utils.data.DataLoader) Pytorch dataloader containing data to test.
    :param model: (torch.nn.Module) The prediction model.
    :param device: (torch.device) The CUDA or CPU device to parallelize.
    :param num_classes: (int) Number of classes of the classification problem (Default = 34).

    :return: metric_result: Dictionary containing the metric results:
                            mean_acc, tot_acc, conf_matrix, mean_AP, AP_list
    """
    # Put model in evaluation mode
    model.eval()

    # Global true tensor
    glob_true = np.array([])
    # Global probability tensor
    glob_prob = np.array([])

    count = 1
    # Computing loop
    with tqdm(loader, unit="batch") as t_loader:
        for data in t_loader:  # Iterate in batches over the training/test dataset.
            t_loader.set_description(f"Batch {count}")
            # Get the inputs of the model (x) and the groundtruth (y)
            input_x, input_y = data
            input_x, input_y = input_x.to(device), input_y.to(device)
            # Get the model output
            out = model(input_x)
            # Get probabilities
            prob = out.softmax(dim=1).cpu().detach().numpy()  # Finds probability for all cases
            true = input_y.cpu().numpy()
            # Stack cases with previous ones
            glob_prob = np.vstack([glob_prob, prob]) if glob_prob.size else prob
            glob_true = np.hstack((glob_true, true)) if glob_true.size else true
            # Update counter
            count += 1

    # Results dictionary declaration
    metric_result = {}
    # Get predictions
    pred = glob_prob.argmax(axis=1)
    # Get confusion matrix
    conf_matrix = sklearn.metrics.confusion_matrix(glob_true, pred, labels=np.arange(num_classes))
    # Normalize confusion matrix by row
    row_norm_conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)
    # Compute mean recall
    mean_acc = np.mean(np.diag(row_norm_conf_matrix))
    # Whole correctly classified cases
    correct = np.sum(np.diag(conf_matrix))
    tot_acc = correct / len(loader.dataset)
    # Get binary GT matrix
    # Handle a binary problem because sklearn.preprocessing.label_binarize gives a matrix with only one column and two are needed
    if num_classes == 2:
        binary_gt = np.zeros((glob_true.shape[0], 2))
        binary_gt[np.arange(glob_true.shape[0]), glob_true] = 1
        # Cast binary_gt to int
        binary_gt = binary_gt.astype(int)
        # If the problem is binary the PR curve and max f1 are obtained for the positive class
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(glob_true, glob_prob[:, 1])
        metric_result["pr_curve"] = precision, recall, thresholds
        # Compute max F1 score
        metric_result["max_f1"] = np.nanmax(2 * (precision * recall) / (precision + recall))
    else:
        binary_gt = sklearn.preprocessing.label_binarize(glob_true, classes=np.arange(num_classes))
    
    AP_list = sklearn.metrics.average_precision_score(binary_gt, glob_prob, average=None)
    mean_AP = np.mean(AP_list)
    
    # Compute the probabilities of taking the correct decision
    correct_prob = glob_prob[np.arange(len(glob_prob)), glob_true]
    # Make a dataframe of correct_prob and true label
    correct_prob_df = pd.DataFrame({'lab_num': glob_true, 'correct_prob': correct_prob})


    # Assign results
    metric_result['mean_acc'] = mean_acc
    metric_result['tot_acc'] = tot_acc
    metric_result['conf_matrix'] = conf_matrix
    metric_result['mean_AP'] = mean_AP
    metric_result['AP_list'] = AP_list
    metric_result['correct_prob_df'] = correct_prob_df

    return metric_result


def print_both(p_string, f):
    """
    This function prints p_string in terminal and to a .txt file with handle f 

    Parameters
    ----------
    p_string : str
        String to be printed.
    f : file
        Txt file handle indicating where to print. 
    """
    print(p_string)
    # f.write(p_string)
    # f.write('\n')
    print(p_string, file=f)
    print('\n', file=f)


def print_epoch(train_dict, test_dict, loss, epoch, fold, path):
    """
    This function prints in terminal a table with all available metrics in all test groups (train, test) for an specific epoch.
    It also writes this table to the training log specified in path.
    :param train_dict: (Dict) Dictionary containing the train set metrics according to the test() function.
    :param test_dict: (Dict) Dictionary containing the test set metrics according to the test() function.
    :param loss: (float) Mean epoch loss value.
    :param epoch: (int) Epoch number.
    :param path: (str) Training log path.
    """
    rows = ['Train', 'Test']
    data = np.zeros((2, 1))
    headers = []
    counter = 0

    # Construction of the metrics table
    for k in train_dict.keys():
        # Handle metrics that cannot be printed
        if (k == 'conf_matrix') or (k == 'AP_list') or (k == 'epoch') or (k == 'pr_curve') or (k == 'correct_prob_df'):
            continue
        headers.append(k)

        if counter > 0:
            data = np.hstack((data, np.zeros((2, 1))))

        data[0, counter] = train_dict[k]
        data[1, counter] = test_dict[k]
        counter += 1

    data_frame = pd.DataFrame(data, index=rows, columns=headers)

    # Print metrics to console and log
    with open(path, 'a') as f:
        print_both('-'*100, f)
        print_both('\n', f)
        print_both(f'Fold {fold}, Epoch {epoch+1}:', f)
        print_both(f'Loss = {loss.cpu().detach().numpy()}', f)
        print_both('\n', f)
        print_both(data_frame, f)
        print_both('\n', f)
    

def print_final_performance(fold_performance, path):

    # Declare the invalid metrics for not considering them
    invalid_metrics = ['conf_matrix', 'AP_list', 'epoch', 'pr_curve', 'correct_prob_df']
    # Obtain a list of the metric dicts for test in the last epoch
    final_test_list = [fold_performance[fold]['test'][-1] for fold in fold_performance.keys()]
    # Obtain a list of the valid metrics
    valid_metrics = [metric for metric in final_test_list[0].keys() if not(metric in invalid_metrics)]
    # Declare empty metric matrix. It has first all the folds data and then the mean and std
    metric_matrix = np.zeros((len(final_test_list), len(valid_metrics)))

    # Assign data of each fold
    for i in range(len(final_test_list)):
        for j in range(len(valid_metrics)):
            metric_matrix[i,j] = final_test_list[i][valid_metrics[j]]
    
    # Get a matrix of various statistics of the folds
    statistic_matrix = np.vstack((  np.ndarray.min(metric_matrix, axis=0, keepdims=True),
                                    np.ndarray.max(metric_matrix, axis=0, keepdims=True),
                                    np.mean(metric_matrix, axis=0, keepdims=True),
                                    np.std(metric_matrix, axis=0, keepdims=True)))
    
    # Obtain index and matrix for print data frame
    index = [f'Fold {i+1}' for i in range(len(final_test_list))]
    index.extend(['Min', 'Max', 'Mean', 'Std'])

    metric_stats_matrix = np.vstack((metric_matrix, statistic_matrix))
    
    # Define printing dataframe
    print_df = pd.DataFrame(metric_stats_matrix, index=index, columns=valid_metrics)
    print_df.index.name = 'Measure'

    # Get the final epoch whose results we are plotting
    final_epoch = len(fold_performance[0]['test'])

    # Open log file and print
    with open(path, 'a') as f:
        print_both('-'*100, f)
        print_both('\n', f)
        print_both(f'General results at epoch {final_epoch}:', f)
        print_both(print_df, f)
  

def get_paths(exp_name):
    results_path = os.path.join("results", exp_name)
    path_dict = {'results': results_path,
                 'train_log': os.path.join(results_path, "training_log.txt"),
                 'metrics': os.path.join(results_path, "metric_dicts.pickle"),
                 'train_fig': os.path.join(results_path, "training_performance.png"),
                 'conf_matrix_fig': os.path.join(results_path, "confusion_matrix"),
                 'violin_conf_fig': os.path.join(results_path, "violin_confidence.png"),
                 'pr_fig': os.path.join(results_path, "pr_curves.png"),
                 'rankings': os.path.join('rankings'),
                 '1_ranking': os.path.join('rankings','1_candle_ranking.csv'),
                 'figures': os.path.join('figures'),
                 'weights_demo_fig': os.path.join('figures','random_class_weights_plot.png')}
    
    return path_dict


def plot_training(fold_performance, save_path):
    """
    FIXME: Update documentation
    This function plots a 2X1 figure. The left figure has the training performance in train and val. The right figure has
    the evolution of the mean training loss over the epochs.
    :param train_list: (dict list) List containing the train metric dictionaries according to the test() function. One
                        value per epoch.
    :param val_list: (dict list) List containing the val metric dictionaries according to the test() function. One
                        value per epoch.
    :param loss: (list) Training loss value list. One value per epoch.
    :param save_path: (str) The path to save the figure.
    """

    global_folds_dict = {}
    for key in fold_performance.keys():
        fold_train_list = fold_performance[key]['train']
        fold_test_list = fold_performance[key]['test']
        fold_loss_list = fold_performance[key]['loss']

        fold_dict = {'Train mACC': [], 'Train mAP': [], 'Test mACC': [], 'Test mAP': [], 'Loss':[]}

        for i in range(len(fold_loss_list)):
            fold_dict['Loss'].append(float(fold_loss_list[i]))
            fold_dict['Train mACC'].append(fold_train_list[i]['mean_acc'])
            fold_dict['Train mAP'].append(fold_train_list[i]['mean_AP'])
            fold_dict['Test mACC'].append(fold_test_list[i]['mean_acc'])
            fold_dict['Test mAP'].append(fold_test_list[i]['mean_AP'])

        global_folds_dict[key] = fold_dict

    global_folds_dict = {(innerKey, outerKey+1): values for outerKey, innerDict in global_folds_dict.items() for innerKey, values in innerDict.items()}
    
    plotting_df = pd.DataFrame(global_folds_dict)
    plotting_df.index += 1
    plotting_df.index.name = 'Epoch'

    handles = [Line2D([0], [0], color='k', label='Train'), Line2D([0], [0], color='darkcyan', label='Test')]
    fig, ax = plt.subplots(1, 3)
    plotting_df['Train mACC'].plot(ax=ax[0], grid=True, xlabel='Epochs', ylabel='Balanced Accuracy', color='k', xlim=[1, len(plotting_df)], ylim=[None,1], legend=False)
    plotting_df['Test mACC'].plot(ax=ax[0], grid=True, xlabel='Epochs', ylabel='Balanced Accuracy', color='darkcyan', xlim=[1, len(plotting_df)], ylim=[None,1], legend=False)
    plotting_df['Train mAP'].plot(ax=ax[1], grid=True, xlabel='Epochs', ylabel='Mean Average Precision', color='k', xlim=[1, len(plotting_df)], ylim=[None,1], legend=False)
    plotting_df['Test mAP'].plot(ax=ax[1], grid=True, xlabel='Epochs', ylabel='Mean Average Precision', color='darkcyan', xlim=[1, len(plotting_df)], ylim=[None,1], legend=False)
    plotting_df['Loss'].plot(ax=ax[2], grid=True, xlabel='Epochs', ylabel='Loss', color='k', xlim=[1, len(plotting_df)], ylim=[0,None], legend=False)
    ax[0].legend(handles=handles)
    ax[1].legend(handles=handles)
    [(axis.spines.right.set_visible(False), axis.spines.top.set_visible(False)) for axis in ax]
    fig.suptitle('Model Training Performance', fontsize=20)
    fig.set_size_inches((18,5))
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)


def plot_conf_matrix(fold_performance, lab_txt_2_lab_num, save_path):
    """
    Plots a heatmap for all the important confusion matrices (train, test). All matrices enter as a
    numpy array.
    :param train_conf_mat: (numpy array) Training confusion matrix.
    :param test_conf_mat: (numpy array) Test confusion matrix.
    :param lab_txt_2_lab_num: (dict) Dictionary that maps the label text to the label number for this dataset.
    :param save_path: (str) General path of the experiment results folder.
    """

    # Handle binary problem when plotting confusion matrix
    if (len(set(lab_txt_2_lab_num.values())) == 2) and (len(lab_txt_2_lab_num.keys()) > 2):
        binary_problem = True
        classes = [0, 1]
    else:
        binary_problem = False
        lab_num_2_lab_txt = {v: k for k, v in lab_txt_2_lab_num.items()}
        class_values_list = sorted(list(lab_txt_2_lab_num.values()))
        classes = [lab_num_2_lab_txt[lab] for lab in class_values_list]

    # Add all the test confusion matrices from each fold
    glob_conf_mat = sum([fold_dict['test'][-1]['conf_matrix'] for fold_dict in fold_performance.values()])


    # Define dataframes
    conf_mat_df = pd.DataFrame(glob_conf_mat, classes, classes)
    # If a NaN resulted from the division a -1 is inserted
    p_df = round(conf_mat_df.div(conf_mat_df.sum(axis=0), axis=1), 2).fillna(-1)
    r_df = round(conf_mat_df.div(conf_mat_df.sum(axis=1), axis=0), 2).fillna(-1)
    # Plot params
    scale = 1.5 if binary_problem==False else 3.0
    fig_size = (50, 30)
    tit_size = 60
    lab_size = 30
    
    d_colors = ["white", "darkcyan"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", d_colors)

    # Plot global confusion matrix
    plt.figure(figsize=fig_size)
    sn.set(font_scale=scale)
    ax = sn.heatmap(conf_mat_df, annot=True, linewidths=.5, fmt='g', cmap=cmap1, linecolor='k', norm=colors.LogNorm(vmin=0.1, vmax=1000))
    plt.title("Confusion Matrix", fontsize=tit_size)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    ax.tick_params(labelsize=lab_size)
    plt.xlabel("Predicted", fontsize=tit_size)
    plt.ylabel("Groundtruth", fontsize=tit_size)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=lab_size)
    cbar.ax.set_ylabel('Number of Samples', fontsize=tit_size)
    plt.tight_layout()
    plt.savefig(save_path+".png", dpi=200)
    plt.close()

    # Plot precision matrix
    plt.figure(figsize=fig_size)
    ax = sn.heatmap(p_df, annot=True, linewidths=.5, fmt='g', cmap=cmap1, linecolor='k', vmin=0.0, vmax=1)
    plt.title("Precision Matrix", fontsize=tit_size)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    ax.tick_params(labelsize=lab_size)
    plt.xlabel("Predicted", fontsize=tit_size)
    plt.ylabel("Groundtruth", fontsize=tit_size)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=lab_size)
    cbar.ax.set_ylabel('Precision', fontsize=tit_size)
    plt.tight_layout()
    plt.savefig(save_path + "_p.png", dpi=200)
    plt.close()
    
    # Plot recall matrix
    plt.figure(figsize=fig_size)
    ax = sn.heatmap(r_df, annot=True, linewidths=.5, fmt='g', cmap=cmap1, linecolor='k', vmin=0.0, vmax=1)
    plt.title("Recall Matrix", fontsize=tit_size)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    ax.tick_params(labelsize=lab_size)
    plt.xlabel("Predicted", fontsize=tit_size)
    plt.ylabel("Groundtruth", fontsize=tit_size)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=lab_size)
    cbar.ax.set_ylabel('Recall', fontsize=tit_size)
    plt.tight_layout()
    plt.savefig(save_path + "_r.png", dpi=200)
    plt.close()

def plot_confidence_violin(fold_performance, lab_txt_2_lab_num, save_path):
    
    # Get a list of all the correct probabilities dataframes
    confidence_df_list = [fold_performance[fold]['test'][-1]['correct_prob_df'] for fold in fold_performance.keys()]
    # Concatenate all folds in a global dataframe
    global_confidence_df = pd.concat(confidence_df_list, ignore_index=True)
    # Reverse lab_txt_2_lab_num dict
    lab_num_2_lab_txt = {v: k for k, v in lab_txt_2_lab_num.items()}
    # Obtain a column with all textual labels
    global_confidence_df['lab_txt'] = global_confidence_df['lab_num'].map(lab_num_2_lab_txt)

    # Code to compute the ordering of the violins
    median_ordered_confidence_df = global_confidence_df.groupby('lab_txt').median().sort_values('correct_prob', ascending=False)
    # Get ordered TCGA and GTEx
    ordered_tcga = median_ordered_confidence_df.index[median_ordered_confidence_df.index.str.contains('TCGA')]
    ordered_gtex = median_ordered_confidence_df.index[median_ordered_confidence_df.index.str.contains('GTEX')]
    # Join both orders in a global order
    lab_order = ordered_tcga.append(ordered_gtex)

    d_colors = ["white", "darkcyan"]
    cmap = LinearSegmentedColormap.from_list("mycmap", d_colors)
    norm = colors.LogNorm(vmin=5, vmax=1000)
    sample_count_df = global_confidence_df['lab_txt'].value_counts()
    color_df= sample_count_df.apply(lambda x: cmap(norm(x)))
    palette = color_df.to_dict()

    fig, ax = plt.subplots(1,1)
    sn.violinplot(data=global_confidence_df, x='lab_txt', y='correct_prob', ax=ax, cut=0, scale='width', order = lab_order, inner='box', palette=palette)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    ax.set_xlabel(None)
    ax.set_ylabel('Confidence Score')
    ax.set_title('Confidence Scores per Class')
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array([])
    cbar = plt.colorbar(m, ax=ax, label='Number of Samples', aspect= 20, pad=0.02)
    cbar.ax.tick_params(labelsize=15)
    ax = cbar.ax
    fig.set_size_inches((15, 7))
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)


def plot_pr_curve(pr_curve_train, pr_curve_val, save_path):
    precision = {'train': pr_curve_train[0], 'val': pr_curve_val[0]}
    recall = {'train': pr_curve_train[1], 'val': pr_curve_val[1]}

    plt.figure(figsize=(11, 10))

    f_scores = np.linspace(0.1, 0.9, num=9)
    for f_score in f_scores:
        x = np.linspace(0.01, 1.01, 499)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("$F_1={0:0.1f}$".format(f_score), xy=(0.9, y[450] + 0.02))

    plt.plot(recall['train'], precision['train'], color='darkorange', lw=2, label='Train')
    plt.plot(recall['val'], precision['val'], color='navy', lw=2, label='Val')
    plt.xlabel('Recall', fontsize=24)
    plt.ylabel('Precision', fontsize=24)
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    plt.title('Precision-Recall Curve', fontsize=28)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.tick_params(labelsize=15)
    plt.savefig(save_path, dpi=200)
    plt.close()
