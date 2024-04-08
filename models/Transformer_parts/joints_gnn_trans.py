import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os ########################
# from  torch.nn import Linear as GCN ######remove this later

from trans_pos_encode import TT_Pos_Encode
from interact import Interact
from gcn_model import GCN
from mlp import MLP
from Transformer_parts.transformer_no_pool import initialize_weight
from Transformer_parts.transformer_no_pool import Transformer as Transformer_joints

class Joints_GNN_Trans(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, in_dim, hidden_size, mlp_numbers, num_layers, num_nodes=25, new_strategy=False, democracy= False):
        super(Joints_GNN_Trans, self).__init__()

        self.N = num_nodes
        ########Position Encoding
        N,d = 25,8
        PE = TT_Pos_Encode(hidden_size, N, d)
        pos_encode = PE.get_position_encoding(need_plot=True)
        self.register_buffer('pos_encode', pos_encode)
        self.proj_input = nn.Linear(in_dim,hidden_size,bias=False)
        in_dim = hidden_size
        ######################################

        ########Interaction
        self.interaction = Interact(in_dim, hidden_size)
        ######################################

        time_dim, num_classes = mlp_numbers
        self.democracy = democracy

        self.gnn_model = nn.ModuleList()
        self.transformer = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout_conv = nn.ModuleList()
        self.convert = nn.ModuleList() #for skip connection (Bias is needed)

        self.num_layers = num_layers
        for i in range(num_layers):
            if new_strategy:
                self.gnn_model.append(GCN(in_dim, hidden_size))
                self.convert.append(nn.Identity())
            else:
                self.gnn_model.append(GCN(in_dim, hidden_size) if i==0 else GCN(hidden_size, hidden_size))
                self.convert.append( nn.Linear(in_dim,hidden_size) if i==0 else nn.Identity()) #for skip connection (Bias is needed)
            self.transformer.append(Transformer_joints(hidden_size=num_nodes*hidden_size,filter_size=num_nodes*hidden_size))
            # self.batch_norm.append(nn.BatchNorm2d(hidden_size))
            self.dropout_conv.append(nn.Dropout(0.1))

        # consider changing from (num_nodes, 1) to (num_nodes, q) # where q<=3
        q = 1 #q decides how much we pool the final layer
        self.pool_result = nn.Linear(num_nodes, q) #for pooling the final layer

        # self.k3_cnn = nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=3,stride=3)
        # self.k5_cnn = nn.Conv1d(in_channels=hidden_size,out_channels=hidden_size//2,kernel_size=5)
        # self.k3_max = nn.MaxPool1d(kernel_size=3,stride=3)

        self.pool_batchnorm = nn.BatchNorm1d(q*hidden_size*time_dim)
        self.fc = MLP(num_layers=1, input_dim=q*hidden_size*(time_dim), hidden_dim=1, output_dim=num_classes)

        if not new_strategy:
            initialize_weight(self.convert[0])
        initialize_weight(self.pool_result)



    def forward(self, features, A, reached_starting_epoch=False):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """

        features = self.proj_input(features)
        if self.pos_encode is not None:
            b, _, _, _ = features.shape
            # we use .detach() for the position encoding
            # so it is not part of the computation graph when computing gradients.
            pos_encode = self.pos_encode.detach()
            features = features + pos_encode.repeat(2,1) #Add position encoding

        features = self.interaction(features[:,:,:self.N], features[:,:,self.N:])
        
        # Layer 1
        last_out = features
        out = features
        for i in range(self.num_layers):
            out = self.gnn_model[i](out , A) #[b x time x n_nodes x filter_size]
            out = self.transformer[i](out)
            last_out = out + self.convert[i](last_out) # Skip connection

            # #reshape for batch_norm
            # out = last_out.transpose(2,3) #[b x time x filter_size x n_nodes]
            # out = out.transpose(1,2) #[b x filter_size x time x n_nodes]
            # out = self.batch_norm[i](out)
            # # undo reshape
            # out = out.transpose(1,2) #[b x time x filter_size x n_nodes]
            # out = out.transpose(2,3) #[b x time x n_nodes x filter_size]
            # # reshape done
            out = F.relu(last_out)
            out = self.dropout_conv[i](out) #dropout #[b x time x n_nodes x filter_size]

        candidates = None
        if self.democracy and reached_starting_epoch and self.training:
            candidates = out #[b x time x n_nodes x filter_size]

        out = self.pool_result(out.transpose(2,3)).squeeze(-1) #[b x time x filter_size x q] #pooling
        # out = self.k3_cnn(out.transpose(2,1)).transpose(2,1) #[b x time//3 x filter_size]
        out = out.flatten(1,-1) #[b x q*time*filter_size]
        out = self.pool_batchnorm(out)
        out = F.relu(out)

        if not self.democracy:
            return self.fc(out) #[b x num_class]
        else:
            return self.fc(out), candidates
