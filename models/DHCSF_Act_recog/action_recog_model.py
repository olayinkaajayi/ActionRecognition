import torch
import torch.nn as nn
import sys
from torch.nn import init

# sys.path.append("DHCS_implement/models")
from stacked_DHCSF_p_TCN import Stack_DHCSF
from mlp import MLP

class Action_recognition(nn.Module):
    """This module implements the skeletal action recognition model proposed in the paper:
        Learning Skeletal Graph Neural Networks for Hard 3D Pose Estimation.
    """

    def __init__(self, mlp_numbers, A,S,L,d,F,num_nodes,in_dim,out_dim_HCSF,conv_out_dim,num_stacks=3,average=True):
        super(Action_recognition, self).__init__()

        self.stacks = Stack_DHCSF(A,S,L,d,F,num_nodes,in_dim,out_dim_HCSF,conv_out_dim,num_stacks,average)
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
