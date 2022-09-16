import sklearn.metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import seaborn as sn
import pandas as pd
from tqdm import tqdm
import matplotlib.colors as colors

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
            input_x, input_y = data.x.to(device), data.y.to(device)

            # Perform a single forward pass.
            out = model(input_x, data.edge_index.to(device), data.batch.to(device)) 
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
    This function calculates mean average precision, mean accuracy, total accuracy and the confusion
    matrix for any classification/detection problem that consists of graph input data. This function can also test an
    adversarial attack on the inputs.
    :param loader: (torch.utils.data.DataLoader) Pytorch dataloader containing data to test.
    :param model: (torch.nn.Module) The prediction model.
    :param device: (torch.device) The CUDA or CPU device to parallelize.
    :param metric: (str) The metric to avaluate the performance can be 'acc', 'mAP' or 'both'.
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
            input_x, input_y = data.x.to(device), data.y.to(device)
            # Get the model output
            out = model(input_x, data.edge_index.to(device), data.batch.to(device))
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

        # Assign results
        metric_result["mean_acc"] = mean_acc
        metric_result["tot_acc"] = tot_acc
        metric_result["conf_matrix"] = conf_matrix
        metric_result["mean_AP"] = mean_AP
        metric_result["AP_list"] = AP_list

    return metric_result


def print_epoch(train_dict, test_dict, loss, epoch, path):
    """
    This function prints in terminal a table with all available metrics in all test groups (train, test, adversarial
    test) for an specific epoch. It also write this table to the training log specified in path.
    :param train_dict: (Dict) Dictionary containing the train set metrics acording to the test() function.
    :param test_dict: (Dict) Dictionary containing the test set metrics acording to the test() function.
    :param loss: (float) Mean epoch loss value.
    :param epoch: (int) Epoch number.
    :param path: (str) Training log path.
    """
    rows = ["Train", "Val"]
    data = np.zeros((2, 1))
    headers = []
    counter = 0

    # Construccion of the metrics table
    for k in train_dict.keys():
        # Handle metrics that cannot be printed
        if (k == "conf_matrix") or (k == "AP_list") or (k == "epoch") or (k == "pr_curve"):
            continue
        headers.append(k)

        if counter > 0:
            data = np.hstack((data, np.zeros((2, 1))))

        data[0, counter] = train_dict[k]
        data[1, counter] = test_dict[k]
        counter += 1

    # Print metrics to console
    print('-----------------------------------------')
    print('                                         ')
    print("Epoch "+str(epoch+1)+":")
    print("Loss = " + str(loss.cpu().detach().numpy()))
    print('                                         ')
    data_frame = pd.DataFrame(data, index=rows, columns=headers)
    print(data_frame)
    print('                                         ')
    # Save metrics to a training log
    with open(path, 'a') as f:
        print('-----------------------------------------', file=f)
        print('                                         ', file=f)
        print("Epoch " + str(epoch + 1) + ":", file=f)
        print("Loss = " + str(loss.cpu().detach().numpy()), file=f)
        print('                                         ', file=f)
        print(data_frame, file=f)
        print('                                         ', file=f)


def plot_training(train_list, val_list, loss, save_path):
    """
    This function plots a 2X1 figure. The left figure has the training performance in train and val. The rigth figure has
    the evolution of the mean training loss over the epochs.
    :param train_list: (dict list) List containing the train metric dictionaries acording to the test() function. One
                        value per epoch.
    :param val_list: (dict list) List containing the val metric dictionaries acording to the test() function. One
                        value per epoch.
    :param loss: (list) Training loss value list. One value per epoch.
    :param save_path: (str) The path to save the figure.
    """
    total_epochs = len(loss)
    
    # Extract mAP data
    data_train_mAP = [metric_dict["mean_AP"] for metric_dict in train_list]
    data_val_mAP = [metric_dict["mean_AP"] for metric_dict in val_list]
    # Extract acc data
    data_train_acc = [metric_dict["mean_acc"] for metric_dict in train_list]
    data_val_acc = [metric_dict["mean_acc"] for metric_dict in val_list]
    # Join data
    data_train = [data_train_acc, data_train_mAP]
    data_val = [data_val_acc, data_val_mAP]
    y_label = "Mean AP / Mean Accuracy"
    legends = ["Train mACC", "Train mAP", "Val mACC", "Val mAP"]
    
    # Generate performance plot
    plt.figure(figsize=(20, 7))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(total_epochs), np.array(data_train).transpose(), '-o')
    plt.plot(np.arange(total_epochs), np.array(data_val).transpose(), '-o')
    plt.grid()
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.title("Model performance", fontsize=25)
    plt.legend(legends)

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(total_epochs), np.array(loss), '-o')
    plt.grid()
    plt.xlabel("Epochs", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.title("Model Loss", fontsize=25)

    plt.savefig(save_path, dpi=200)


def plot_conf_matrix(train_conf_mat, test_conf_mat, lab_txt_2_lab_num, save_path):
    """
    Plots a heatmap for all the important confusion matrices (train, test and adversarial test). All matrices enter as a
    numpy array.
    :param train_conf_mat: (numpy array) Training confusion matrix.
    :param test_conf_mat: (numpy array) Test confusion matrix.
    :param lab_txt_2_lab_num: (dict) Dictionary that maps the label text to the label number for this dataset.
    :param save_path: (str) General path of the experiment results folder.
    """

    # Handle binary problem when ploting confusion matrix
    if (len(set(lab_txt_2_lab_num.values())) == 2) and (len(lab_txt_2_lab_num.keys()) > 2):
        binary_problem = True
        classes = [0, 1]
    else:
        binary_problem = False
        classes = sorted(list(lab_txt_2_lab_num.keys()))

    # Define dataframes
    df_train = pd.DataFrame(train_conf_mat, classes, classes)
    df_test = pd.DataFrame(test_conf_mat, classes, classes)

    # Plot params
    scale = 1.5 if binary_problem==False else 3.0
    fig_size = (50, 30)
    tit_size = 40
    lab_size = 30
    cm_str = 'Purples'

    # Plot confusion matrix for train
    plt.figure(figsize=fig_size)
    sn.set(font_scale=scale)
    ax = sn.heatmap(df_train, annot=True, linewidths=.5, fmt='g', cmap=plt.get_cmap(cm_str),
                    linecolor='k', norm=colors.LogNorm(vmin=0.1, vmax=1000))
    plt.title("Train \nConfusion matrix", fontsize=tit_size)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    ax.tick_params(labelsize=lab_size)
    plt.xlabel("Predicted", fontsize=tit_size)
    plt.ylabel("Groundtruth", fontsize=tit_size)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=lab_size)
    plt.tight_layout()
    plt.savefig(save_path+"_train.png", dpi=200)
    plt.close()

    # Plot confusion matrix for test
    plt.figure(figsize=fig_size)
    sn.set(font_scale=scale)
    ax = sn.heatmap(df_test, annot=True, linewidths=.5, fmt='g', cmap=plt.get_cmap(cm_str),
                    linecolor='k', norm=colors.LogNorm(vmin=0.1, vmax=1000))
    plt.title("Validation\nConfusion matrix", fontsize=tit_size)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    ax.tick_params(labelsize=lab_size)
    plt.xlabel("Predicted", fontsize=tit_size)
    plt.ylabel("Groundtruth", fontsize=tit_size)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=lab_size)
    plt.tight_layout()
    plt.savefig(save_path + "_val.png", dpi=200)
    plt.close()


# TODO: Remove pr_curve_adv_val parameter
def plot_pr_curve(pr_curve_train, pr_curve_val, pr_curve_adv_val, save_path):
    precision = {'train': pr_curve_train[0], 'val': pr_curve_val[0], 'adv_val': pr_curve_adv_val[0]}
    recall = {'train': pr_curve_train[1], 'val': pr_curve_val[1], 'adv_val': pr_curve_adv_val[1]}
    threshold = {'train': pr_curve_train[2], 'val': pr_curve_val[2], 'adv_val': pr_curve_adv_val[2]}

    plt.figure(figsize=(11, 10))

    f_scores = np.linspace(0.1, 0.9, num=9)
    for f_score in f_scores:
        x = np.linspace(0.01, 1.01, 499)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("$f_1={0:0.1f}$".format(f_score), xy=(0.9, y[450] + 0.02))

    plt.plot(recall['train'], precision['train'], color='darkorange', lw=2, label='Train')
    plt.plot(recall['adv_val'], precision['adv_val'], color='cornflowerblue', lw=2, label='Adv. Val')
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
