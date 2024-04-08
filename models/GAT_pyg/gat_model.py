import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self,in_dim, out_dim, drp_out=0.5, in_head=8, out_head=1):
        super(GAT, self).__init__()
        self.dropout = drp_out
        self.conv1 = GATConv(in_dim, out_dim, heads=in_head, dropout=self.dropout)
        self.conv2 = GATConv(out_dim*in_head, out_dim, #concat=False,
                             heads=out_head, dropout=self.dropout)
# increase out_head and remove concat=False
    def forward(self, data):
        """
            data is of type torch_geometric.data.dataset
        """
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x# F.relu(x)
