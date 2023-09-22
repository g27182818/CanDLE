import torch
import numpy as np
import pandas as pd
import sklearn.metrics

def get_metrics(pred_probability, ground_truth):

    # If the inputs are tensors, convert them to numpy arrays
    if torch.is_tensor(pred_probability):
        pred_probability = pred_probability.cpu().detach().numpy()
    if torch.is_tensor(ground_truth):
        ground_truth = ground_truth.cpu().detach().numpy()
    
    # If the inputs are pandas series, or dataframes convert them to numpy arrays
    if isinstance(pred_probability, pd.DataFrame):
        pred_probability = pred_probability.values
    if isinstance(ground_truth, pd.Series):
        ground_truth = ground_truth.values

    # Get the number of unique classes in the ground truth
    num_classes = len(np.unique(ground_truth))

    # Get the predictions from the probabilities
    predictions = np.argmax(pred_probability, axis=1)

    # Get confusion matrix
    conf_matrix = sklearn.metrics.confusion_matrix(ground_truth, predictions, labels=np.arange(num_classes))
    # Normalize confusion matrix by row
    row_norm_conf_matrix = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)
    # Compute mean recall
    mean_acc = np.mean(np.diag(row_norm_conf_matrix))
    # Whole correctly classified cases
    correct = np.sum(np.diag(conf_matrix))
    tot_acc = correct / len(ground_truth)
    
    # Declare the output dictionary
    metric_result = {}

    # Get binary GT matrix
    # Handle a binary problem because sklearn.preprocessing.label_binarize gives a matrix with only one column and two are needed
    if num_classes == 2:
        binary_gt = np.zeros((ground_truth.shape[0], 2))
        binary_gt[np.arange(ground_truth.shape[0]), ground_truth] = 1
        # Cast binary_gt to int
        binary_gt = binary_gt.astype(int)
        # If the problem is binary the PR curve and max f1 are obtained for the positive class
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(ground_truth, pred_probability[:, 1])
        metric_result["pr_curve"] = precision, recall, thresholds
        # Compute max F1 score
        metric_result["max_f1"] = np.nanmax(2 * (precision * recall) / (precision + recall))
        metric_result['pr_df'] = pd.DataFrame({'lab_num': ground_truth, 'positive_prob': pred_probability[:, 1]})
    else:
        binary_gt = sklearn.preprocessing.label_binarize(ground_truth, classes=np.arange(num_classes))
    
    AP_list = sklearn.metrics.average_precision_score(binary_gt, pred_probability, average=None)
    mean_AP = np.mean(AP_list)
    
    # Compute the probabilities of taking the correct decision
    correct_prob = pred_probability[np.arange(len(pred_probability)), ground_truth]
    # Make a dataframe of correct_prob and true label
    correct_prob_df = pd.DataFrame({'lab_num': ground_truth, 'correct_prob': correct_prob})

    # Assign results
    metric_result['mean_acc'] = mean_acc
    metric_result['tot_acc'] = tot_acc
    metric_result['conf_matrix'] = conf_matrix
    metric_result['mean_AP'] = mean_AP
    metric_result['AP_list'] = AP_list
    metric_result['correct_prob_df'] = correct_prob_df

    return metric_result

