import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from parts_gnn_trans import Parts_GNN_Trans
from body_gnn_trans import Body_GNN_Trans
from joints_gnn_trans import Joints_GNN_Trans

class Spatiotemp_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, in_dim, hidden_size, mlp_numbers, num_trans_layers=2):

        super(Spatiotemp_Action_recog, self).__init__()

        # parts
        self.parts = Parts_GNN_Trans(in_dim, hidden_size, mlp_numbers, num_trans_layers=num_trans_layers) #, num_parts=10)

    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        # parts
        out_parts = self.parts(features, A)

        scores = out_parts

        return scores
