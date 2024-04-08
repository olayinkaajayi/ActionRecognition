import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class ImpConGCN(nn.Module):
    """This module implements the vanilla GCN model."""

    def __init__(self, in_dim, out_dim, use_bn=False, layer1=False):
        super(ImpConGCN, self).__init__()

        self.layer1 = layer1
        self.Wq = nn.Linear(in_dim,out_dim,bias=False)
        self.Wk = nn.Linear(in_dim,out_dim,bias=False)
        self.conv = nn.Linear(out_dim,out_dim,bias=False)

        init.xavier_uniform_(self.conv.weight)
        init.xavier_uniform_(self.Wq.weight)
        init.xavier_uniform_(self.Wk.weight)

        self.use_bn = use_bn
        if self.use_bn: #Consider using Graph Norm here or instance norm
            self.bn = nn.BatchNorm2d(out_dim)


    def learn_adj(self, features, A):
        """We implement the algorithm that learns the adjacency matrix"""

        # May remove normalize
        x1 = self.Wq(features) #F.normalize(self.Wq(features), dim=-2) #[b x time x n_nodes x out_channel]
        x2 = self.Wk(features) #F.normalize(self.Wk(features), dim=-2) #[b x time x n_nodes x out_channel]

        x1x2 = torch.matmul(x1,x2.transpose(-1,-2)) #[b x time x n_nodes x n_nodes]
        out = torch.tanh(x1x2) #[b x time x n_nodes x n_nodes]

        b, t, _, _ = features.shape
        if self.layer1:

            A = torch.from_numpy(A).unsqueeze(0).unsqueeze(0) #[1 x 1 x n_nodes x n_nodes]
            A = A.repeat(b, t, 1, 1).to(features.device) #[b x time x n_nodes x n_nodes]

            mask = torch.logical_not(A)
            # The ReLU will set -ve to 0
            out = F.relu(mask*out) + A #[b x time x n_nodes x n_nodes]
        else:
            # We do not know the adjacency matrix at this layer,
            # so we do not add the original adjacency matrix.
            # But we need the embedding of each node to update itself.
            I = torch.eye(A.shape[0]).unsqueeze(0).unsqueeze(0) #[1 x 1 x n_nodes x n_nodes]
            I = I.repeat(b, t, 1, 1).to(features.device) #[b x time x n_nodes x n_nodes]

            mask = torch.logical_not(I)
            # The ReLU will set -ve to 0
            out = F.relu(mask*out) + I #[b x time x n_nodes x n_nodes]

        return out.float()


    def forward(self, features, A=None, raw_feat=None):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        A = A + np.eye(A.shape[0]) #Add an identity to A for self-loop
        A = self.learn_adj(raw_feat if raw_feat is not None else features, A) #[b x time x n_nodes x n_nodes]

        out = self.conv(features) #[b x time x n_nodes x out_channel]
        out = A.matmul(out).transpose(-1,-2) #[b x time x out_channel x n_nodes]

        if self.use_bn:
            out = self.bn(out.transpose(1,2)) #[b x out_channel x time x n_nodes]
            out = out.transpose(1,2)#[b x time x out_channel x n_nodes]

        return out.transpose(-1,-2) #[b x time x n_nodes x out_channel]
