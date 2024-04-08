import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from  linear_underhood import MyLinear as GCN ######remove this later

from gcn_model import GCN
from gnn_trans_block import GNN_Trans
from mlp import MLP
from transformer import initialize_weight

class Spatiotemp_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, in_dim, mlp_numbers, num_trans_layers):
        super(Spatiotemp_Action_recog, self).__init__()
        time_dim , num_classes = mlp_numbers
        num_nodes = 25 #change to 25 later
        first = 16
        second = 16
        # Layer 1
        # self.gnn_transformer1 = GNN_Trans(GCN(in_dim,first), first*num_nodes, first*num_nodes)
        # self.convert1 = nn.Linear(in_dim,first) #for skip connection (Bias is needed)
        # self.batch_norm1 = nn.BatchNorm2d(first)
        # self.dropout_conv1 = nn.Dropout(0.1)
        self.gnn_transformer1 = nn.ModuleList()
        self.convert1 = nn.ModuleList() #for skip connection (Bias is needed)
        self.batch_norm1 = nn.ModuleList()
        self.dropout_conv1 = nn.ModuleList()
        self.num_trans_layers = num_trans_layers - 1 #This is because we have written by hand the last layer

####convert1 is needed only once. Remove it from ModuleList
        for i in range(self.num_trans_layers):
            self.gnn_transformer1.append( GNN_Trans(GCN(in_dim,first) if i==0 else GCN(first,first), first*num_nodes, first*num_nodes))
            self.convert1.append( nn.Linear(in_dim,first) if i==0 else nn.Identity()) #for skip connection (Bias is needed)
            self.batch_norm1.append( nn.BatchNorm2d(first))
            self.dropout_conv1.append( nn.Dropout(0.1))
            if i == 0:
                initialize_weight(self.convert1[i])

        # Layer 2
####convert2 is not needed since both would be the same size
        self.gnn_transformer2 = GNN_Trans(GCN(first,second), second*num_nodes, second*num_nodes)
        self.convert2 = nn.Identity()# nn.Linear(first,second) #for skip connection (Bias is needed)
        self.batch_norm2 = nn.BatchNorm1d(second*time_dim)
        self.dropout_conv2 = nn.Dropout(0.1)

        self.pool_result = nn.Linear(num_nodes, 1)#, bias=False) #consider making bias False

        # initialize_weight(self.convert1)
        # initialize_weight(self.convert2) #ADDING convert2 MADE NO DIFFERENCE (So it might be the batch_size that impoves the result)
        initialize_weight(self.pool_result)

        self.fc = MLP(num_layers=1, input_dim=second*time_dim, hidden_dim=1, output_dim=num_classes)




    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(batch_size,time,num_nodes,num_nodes)
        """

        # Layer 1
        last_out = features
        out = features
        for i in range(self.num_trans_layers):
            out = self.gnn_transformer1[i](out , A) #[b x time x n_nodes x filter_size]
            last_out = out + self.convert1[i](last_out) # Skip connection

            #reshape for batch_norm
            out = last_out.transpose(2,3) #[b x time x filter_size x n_nodes]
            out = out.transpose(1,2) #[b x filter_size x time x n_nodes]
            out = self.batch_norm1[i](out)
            # undo reshape
            out = out.transpose(1,2) #[b x time x filter_size x n_nodes]
            out = out.transpose(2,3) #[b x time x n_nodes x filter_size]
            # reshape done

            out = F.relu(out)
            out = self.dropout_conv1[i](out) #dropout #[b x time x n_nodes x filter_size]

            # if i != (self.num_trans_layers-1): #NO DIFFERENCE AFTER CHANGE IS MADE
            #     last_out = out ###THIS IS CONTROVERSIAL

        # Layer 2
        out = self.gnn_transformer2(out , A) #[b x time x n_nodes x filter_size]
        last_out = out + self.convert2(last_out) # Skip connection (no need to project)

        # consider doing the pooling with sum or learned pooling
        out = self.pool_result(last_out.transpose(2,3)).squeeze(-1) #[b x time x filter_size] #pooling
        # out = out.sum(-2)
        out = out.flatten(1,2) #[b x time*filter_size]
        #consider removing the next 3 lines
        out = self.batch_norm2(out)
        out = F.relu(out)
        out = self.dropout_conv2(out) #dropout

        return self.fc(out)
