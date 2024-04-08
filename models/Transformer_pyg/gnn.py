import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv

class GAT(nn.Module):
    def __init__(self,in_dim, out_dim, drp_out=0.1, in_head=8, out_head=1):
        super(GAT, self).__init__()
        # out_dim = out_dim//in_head
        self.dropout = drp_out
        self.conv1 = GCNConv(in_dim, out_dim,bias=False)
        self.conv2 = GCNConv(out_dim, out_dim,bias=False)
        # self.conv1 = GATConv(in_dim, out_dim, heads=in_head, dropout=self.dropout)
        # self.conv2 = GATConv(out_dim*in_head, out_dim, concat=False,
        #                      heads=out_head, dropout=self.dropout)

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
        return x
