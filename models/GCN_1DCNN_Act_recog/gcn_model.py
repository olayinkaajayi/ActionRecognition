import torch
import torch.nn as nn
from torch.nn import init

class GCN(nn.Module):
    """This module implements the vanilla GCN model."""

    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()

        self.relu = nn.ReLU()
        if out_dim == 0:
            out_dim = in_dim
        self.W = nn.Parameter(torch.FloatTensor(in_dim,out_dim))
        init.xavier_uniform_(self.W)

    def forward(self, features, A):
        """
            features:   --dim(batch_size,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        batch_size = features.size(0)
        out = torch.bmm( A.unsqueeze(0).repeat(batch_size,1,1) , features )
        return self.relu( torch.bmm( out , self.W.unsqueeze(0).repeat(batch_size,1,1) ) )
