import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, ChebConv

# ChebConv class model, initial baseline
class BaselineModel(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, out_size):
        """
        Class constructor for baseline using ChebConv, Inherits from torch.nn.Module
        :param hidden_channels: (Int) Hidden channels in every layer.
        :param out_size: (Int) Number of output classes.
        """
        super(BaselineModel, self).__init__()
        # Class atributes
        self.hidd = hidden_channels
        self.input_size = input_size
        # Convolution definitions
        self.conv1 = ChebConv(1, hidden_channels, K=5)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K=5)
        self.conv3 = ChebConv(hidden_channels, hidden_channels, K=5)
        self.lin = Linear(hidden_channels * self.input_size, out_size)

    def forward(self, x, edge_index, batch):
        """
        Performs a forward pass.
        :param x: (torch.Tensor) Input features of each node.
        :param edge_index: (torch.Tensor) Edges indicating graph connectivity.
        :param batch: (torch.Tensor) Batch vector indicating the correspondence of each node in the batch.
        :return: (torch.Tensor) Matrix of logits, each row corresponds with a patient in the batch and each column represent a
                 cancer or normal type logit.
        """
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.lin(torch.reshape(x, (torch.max(batch).item() + 1, self.input_size * self.hidd)))
        return x


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

    def forward(self, x, edge_index, batch):
        """
        Performs a forward pass of the MLP model. To provide coherence in the training and etsting, this function asks
        for edge indices. However, this parameter is ignored.
        :param x: (torch.Tensor) Input features of each node.
        :param edge_index: (torch.Tensor) Edges indicating graph connectivity. This parameter is ignored.
        :param batch: (torch.Tensor) Batch vector indicating the correspondence of each node in the batch. Just used for a
                      reshape.
        :return: (torch.Tensor) Matrix of logits, each row corresponds with a patient in the batch and each column represent a
                 cancer or normal type logit.
        """
        # Resahpe x
        x = torch.reshape(x, (torch.max(batch).item() + 1, self.hidd_sizes[0]))
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






