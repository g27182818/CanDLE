import torch
import torch.nn.functional as F
import torch.nn as nn


# MLP module for simple comparison
class MLP(torch.nn.Module):
    def __init__(self, h_sizes, out_size, act="relu"):
        """
        Class constructor for simple comparison, Inherits from torch.nn.Module. This model DOES NOT include graph
        connectivity information or any other. It uses raw input.
        :param h_sizes: (list) List of sizes of the hidden layers. Does not include the output size.
        :param out_size: (int) Number of output classes.
        :param act: (str) Paramter to specify the activation function. Can be "relu", "sigmoid" or "gelu". Default
                    "relu" (Default = "relu").
        """
        super(MLP, self).__init__()
        # Activation function definition
        self.activation = act
        # Sizes definition
        self.hidd_sizes = h_sizes
        # Hidden layers
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):
        """
        Performs a forward pass of the MLP model.
        :param x: (torch.Tensor) Input features of each node.
        :return: (torch.Tensor) Matrix of logits, each row corresponds with a patient in the batch and each column represent a
                 cancer or normal type logit.
        """
        # Feedforward
        for layer in self.hidden:
            if self.activation == "relu":
                x = F.relu(layer(x))
            elif self.activation == "gelu":
                x = F.gelu(layer(x))
            elif self.activation == "sigmoid":
                x = F.sigmoid(layer(x))
            else:
                raise NotImplementedError("Activation function not impemented")

        # Output layer. This is the only one used in multinomial logistic regression
        output = self.out(x)
        return output

# Define Multitask Hong Model
class HongMultiTask(torch.nn.Module):
    def __init__(self, input_size):
        """
        This is an approximate re-implementation of the desease stage and tissue type model by Hong et al in doi: 10.1038/s41598-022-13665-5 
        "A deep learning model to classify neoplastic state and tissue origin from transcriptomic data" In this case the multitask
        desease stage head just has 2 classes: cancer and healthy
        """
        super(HongMultiTask, self).__init__()
        self.input_size = input_size

        self.lin1 = nn.Linear(self.input_size, 1832)
        self.lin2 = nn.Linear(1832, 29)
        self.lin3 = nn.Linear(29, 429)
        self.lin4 = nn.Linear(429, 203)
        self.lin5 = nn.Linear(203, 118)
        self.cancer_head = nn.Linear(118, 2)
        self.tissue_head = nn.Linear(118, 30)

    def forward(self, x):

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.lin4(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.lin5(x))
        x = F.dropout(x, p=0.1, training=self.training)

        y_cancer = self.cancer_head(x)
        y_tissue = self.tissue_head(x)

        return y_cancer, y_tissue

# Define Subtype cancer classification Hong Model
class HongSubType(torch.nn.Module):
    def __init__(self, input_size):
        """
        This is an approximate re-implementation of the desease stage and tissue type model by Hong et al in doi: 10.1038/s41598-022-13665-5 
        "A deep learning model to classify neoplastic state and tissue origin from transcriptomic data" In this case the multitask
        desease stage head just has 2 classes: cancer and healthy
        """
        super(HongSubType, self).__init__()
        self.input_size = input_size

        self.lin1 = nn.Linear(self.input_size, 784)
        self.lin2 = nn.Linear(784, 308)
        self.lin3 = nn.Linear(308, 344)
        self.lin4 = nn.Linear(344, 21)
        self.lin5 = nn.Linear(21, 62)
        self.subtype_head = nn.Linear(62, 16)

    def forward(self, x):

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.lin4(x))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.lin5(x))
        x = F.dropout(x, p=0.1, training=self.training)

        y_subtype = self.subtype_head(x)

        return y_subtype






