import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from transformer import Transformer

class GNN_Trans(nn.Module):
    """
        This class would combine the GCN and LSTM modules to form our model for the video data.
    """

    def __init__(self, gnn, hidden_size, filter_size,pool=False):
        super(GNN_Trans, self).__init__()
        self.time_len = 300 #extra (constant) info
        self.num_nodes = 25 #extra (constant) info
        self.gnn = gnn
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.transformer = Transformer(hidden_size=hidden_size,filter_size=filter_size,pool=pool)


    def forward(self,data):
        """
            data:   data type of class torch_geometric.Data.data
        """
        batch_size = data.y.size(0)
        out = self.gnn(data) #[b x out_channel]
        out = self.batch_norm(out) #batch_norm (if needed)
        out = out.reshape(batch_size,self.time_len,self.num_nodes,-1) #[b x time x n_nodes x out_channel]
        #take the result of the last step and pass to the transformer module
        out = F.relu(out)

        out = self.transformer(out) #shape: [b x time x n_nodes x filter_size] if pool else [b x time x filter_size]
        #return final result to the calling model.
        return out
