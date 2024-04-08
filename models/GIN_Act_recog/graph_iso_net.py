import numpy as np
import torch
import torch.nn as nn
from mlp import MLP

class GIN(nn.Module):
    """This module implements the Graph Isomorphism Network."""

    def __init__(self, in_dim , num_layers , hidden_dim , out_dim , eps=0.01):
        super(GIN, self).__init__()
        learn_eps = False if eps>0 else True
        self.eps = nn.Parameter(torch.FloatTensor(1)) if learn_eps else eps
        if out_dim == 0:
            out_dim = in_dim
        self.MLP = MLP(num_layers=num_layers, input_dim=in_dim, hidden_dim=hidden_dim, output_dim=out_dim)

    def forward(self, features , A):
        """
            features:   --dim(batch_size,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        batch_size = features.size(0)
        h = features
        A = A + (1 + self.eps)*np.eye(A.shape[0])
        if features.device.type == 'cuda':
            A = torch.FloatTensor(A).cuda()
        else:
            A = torch.FloatTensor(A).to(features.device)

        out = torch.bmm( A.unsqueeze(0).repeat(batch_size,1,1) , h )
        tmp = [self.MLP(out[:,i]).unsqueeze(1) for i in range(features.size(1))] #num_nodes = features.size(1)
        h = torch.cat(tmp,1)

        return h
