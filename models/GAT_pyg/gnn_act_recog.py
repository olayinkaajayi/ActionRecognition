import torch
import torch.nn as nn
import numpy as np

from gnn_for_frame import GNN_for_frames
from mlp import MLP

class GNN_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, mlp_numbers, num_nodes, in_dim, out_dim, gnn, average=True):
        super(GNN_Action_recog, self).__init__()

        time_dim , num_layers , hidden_dim , output_dim = mlp_numbers
        self.GNN_on_vid_frames = GNN_for_frames(time_dim, num_nodes, gnn, out_dim, average)

        self.fc = MLP(num_layers=num_layers, input_dim=num_nodes*time_dim, hidden_dim=hidden_dim, output_dim=output_dim)


    def forward(self, data):
        """
            data is of type torch_geometric.data.dataset
        """
        out = self.GNN_on_vid_frames(data)
        out = out.flatten(1) #batch_size x num_nodes*time_dim
        return self.fc(out)
