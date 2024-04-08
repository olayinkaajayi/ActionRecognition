from gcn_model import GCN as GCNConv
import numpy as np
import torch
import torch.nn.functional as F


class Body_GNN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers=3):
        super(Body_GNN, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i:
                conv = GCNConv(dim,dim,use_bn=False)
            else:
                conv = GCNConv(num_features, dim, use_bn=False)

            self.convs.append(conv)

    def forward(self, x, adj, get_global=True):
        """
            x:   --dim(batch_size,time,num_nodes,in_channel)
            adj:      --dim(batch_size,num_nodes,num_nodes)
        """
        xs = []
        for i in range(self.num_gc_layers):
            #consider adding the results together using a skip connection
            x = F.relu(self.convs[i](x, adj)) #[b x time x n_nodes x out_channel]
            xs.append(x)

        if get_global:
            xpool = [self.global_add_pool(x) for x in xs] #loop size negligible
            x = torch.cat(xpool, -1)
            return x, torch.cat(xs, -1)
        else:
            # Need this for negative embeddings
            return torch.cat(xs, -1)

    def global_add_pool(self,x):
        """x:   --dim(batch_size,time,num_nodes,in_channel)"""
        return torch.sum(x,dim=-2) #[b x time x out_channel]
