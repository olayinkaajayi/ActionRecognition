import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from parts_gnn import Parts_GNN
from mlp import MLP
from Transformer_parts.transformer_no_pool import initialize_weight
from Transformer_parts.transformer_no_pool_mask import Transformer as Transformer_parts


class Parts_GNN_Trans(torch.nn.Module):
    def __init__(self, in_dim, hidden_size, mlp_numbers, num_trans_layers=2, num_parts=10):
        super(Parts_GNN_Trans, self).__init__()

        time_dim, num_classes = mlp_numbers

        self.gnn_model = Parts_GNN(in_dim, hidden_size, nnext=num_parts)
        self.transformer = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout_conv = nn.ModuleList()

        self.num_trans_layers = num_trans_layers

        for i in range(num_trans_layers):
            self.transformer.append(Transformer_parts(hidden_size=num_parts*hidden_size, filter_size=num_parts*hidden_size, num_nodes=num_parts))
            # self.batch_norm.append(nn.BatchNorm2d(hidden_size))
            self.dropout_conv.append(nn.Dropout(0.1))

        # consider using HIERARCHICAL POOLING here
        # consider changing from (num_nodes, 1) to (num_nodes, q) # where q<=3
        self.pool_result = nn.Linear(num_parts, 1) #for pooling the final layer
        self.pool_batchnorm = nn.BatchNorm1d(hidden_size*time_dim)

        self.fc = MLP(num_layers=1, input_dim=hidden_size*time_dim, hidden_dim=1, output_dim=num_classes)

        initialize_weight(self.pool_result)


    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """

        # out, s_l, anext = self.gnn_model(features,A)
        out = self.gnn_model(features,A)
        out_l = out #used for skip connection
        # Layers
        for i in range(self.num_trans_layers):

            out = self.transformer[i](out) #[b x time x n_nodes x filter_size]

            out_l = out + out_l # Skip connection

            #reshape for batch_norm
            # out = out_l.transpose(2,3) #[b x time x filter_size x n_nodes]
            # out = out.transpose(1,2) #[b x filter_size x time x n_nodes]
            # out = self.batch_norm[i](out)
            # # undo reshape
            # out = out.transpose(1,2) #[b x time x filter_size x n_nodes]
            # out = out.transpose(2,3) #[b x time x n_nodes x filter_size]
            # reshape done
            out = F.relu(out_l)
            out = self.dropout_conv[i](out) #dropout #[b x time x n_nodes x filter_size]

        out = self.pool_result(out.transpose(2,3)).squeeze(-1) #[b x time x filter_size] #pooling
        out = out.flatten(1,2) #[b x time*filter_size]
        out = self.pool_batchnorm(out)
        out = F.relu(out) #may not be needed

        return self.fc(out)#, s_l#, anext
