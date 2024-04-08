import torch
import torch.nn as nn

from gin_stacked import GIN_stack
from mlp import MLP

class GIN_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, mlp_numbers, num_nodes, in_dim, gin_num_layers, gin_hidden_dim, gin_out_dim, eps, num_stacks=3, average=True):
        super(GIN_Action_recog, self).__init__()
        self.stacks = GIN_stack(in_dim, gin_num_layers, gin_hidden_dim, gin_out_dim, eps, num_stacks, average)
        time_dim , num_layers , hidden_dim , output_dim = mlp_numbers
        self.fc = MLP(num_layers=num_layers, input_dim=num_nodes*time_dim, hidden_dim=hidden_dim, output_dim=output_dim)


    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        out = self.stacks(features , A)
        out = out.flatten(1) #batch_size x num_nodes*time_dim

        return self.fc(out)
