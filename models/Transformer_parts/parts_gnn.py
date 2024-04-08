import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from gcn_model import GCN

class Parts_GNN(nn.Module):
    """
        This class would combine the GCN and LSTM modules to form our model for the video data.
    """

    def __init__(self, in_dim, hidden_size, nnext=10): #10 body parts
        super(Parts_GNN, self).__init__()

        self.embed = GCN(in_dim,hidden_size,use_bn=False) #consider use_bn=False
        self.assign_mat = GCN(in_dim, nnext,use_bn=False) #consider use_bn=False

    def forward(self,x,adj):
        """
            x:   --dim(batch_size,time,num_nodes,in_channel)
            adj:      --dim(batch_size,num_nodes,num_nodes)
        """

        z_l = self.embed(x, adj) #[b x time x n_nodes x h_size]
        z_l = F.relu(z_l) #relu
        s_l = self.assign_mat(x, adj) #[b x time x n_nodes x nnext]
        s_l = F.relu(s_l) #relu
        s_l = F.softmax(s_l, dim=-1) #[b x time x n_nodes x nnext]
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l) #[b x time x nnext x h_size]
        # if not ('torch' in str(type(adj))):
        #     adj = torch.from_numpy(adj).float().to(x.device) # moving the numpy array to tensor
        # anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l) #[b x time x nnext x nnext]

        return xnext#, s_l#, anext
