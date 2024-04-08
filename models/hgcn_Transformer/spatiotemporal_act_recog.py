import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from gnn_trans_block import GNN_Trans
from mlp import MLP
from models.base_models import NCModel as HGCN
from transformer import initialize_weight

class Spatiotemp_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, args, in_dim, mlp_numbers):
        super(Spatiotemp_Action_recog, self).__init__()
        time_dim , num_classes = mlp_numbers

        # Layer 1
        self.gnn_transformer1 = GNN_Trans(HGCN(args), 16, 16)
        self.convert1 = nn.Linear(in_dim,16) #for skip connection (Bias is needed)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.dropout_conv1 = nn.Dropout(0.1)

        # Layer 2
        args.feat_dim = 16
        args.dim = 32
        self.gnn_transformer2 = GNN_Trans(HGCN(args), 32, 32)
        self.convert2 = nn.Linear(16,32) #for skip connection (Bias is needed)
        self.batch_norm2 = nn.BatchNorm1d(32*time_dim)
        self.dropout_conv2 = nn.Dropout(0.1)

        num_nodes = 25
        self.pool_result = nn.Linear(num_nodes, 1)#, bias=False) #consider making bias False

        initialize_weight(self.convert1)
        initialize_weight(self.convert2)
        initialize_weight(self.pool_result)

        self.fc = MLP(num_layers=1, input_dim=32*time_dim, hidden_dim=1, output_dim=num_classes)



    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        A = torch.tensor(A).float().to(features.device)

        # Layer 1
        out = self.gnn_transformer1(features , A) #[b x time x n_nodes x filter_size]
        last_out = out + self.convert1(features) # Skip connection

        #reshape for batch_norm
        out = last_out.transpose(2,3) #[b x time x filter_size x n_nodes]
        out = out.transpose(1,2) #[b x filter_size x time x n_nodes]
        out = self.batch_norm1(out)
        # undo reshape
        out = out.transpose(1,2) #[b x time x filter_size x n_nodes]
        out = out.transpose(2,3) #[b x time x n_nodes x filter_size]
        # reshape done

        out = F.relu(out)
        out = self.dropout_conv1(out) #dropout #[b x time x n_nodes x filter_size]

        # Layer 2
        out = self.gnn_transformer2(out , A) #[b x time x n_nodes x filter_size]
        last_out = out + self.convert2(last_out) # Skip connection (no need to project)

        # consider doing the pooling with sum or learned pooling
        out = self.pool_result(out.transpose(2,3)).squeeze(-1) #[b x time x filter_size] #pooling

        out = out.flatten(1,2) #[b x time*filter_size]
        out = self.batch_norm2(out)
        out = F.relu(out)
        out = self.dropout_conv2(out) #dropout

        return self.fc(out)
