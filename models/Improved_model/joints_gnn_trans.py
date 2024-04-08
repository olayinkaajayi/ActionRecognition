import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

###########START WITH JUST POSITION ENCODING##################################################
# from gcn_model_plus_learn import ImpConGCN
from gcn_model import GCN as ImpConGCN
from pos_encode import Position_encode
from mlp import MLP
from transformer import initialize_weight
from transformer import Transformer

class Joints_GNN_Trans(nn.Module):
    """This module implements a skeletal action recognition model using the Improved GCN model
        and a transformer-encoder layer.
    """

    def __init__(self, in_dim, hidden_size, mlp_numbers, N=25, d=8, pos_encode=None):
        super(Joints_GNN_Trans, self).__init__()

        time_dim, num_classes = mlp_numbers

        if pos_encode is not None:
            pos_encode = self.get_position_encoding(N, d)
            self.register_buffer('pos_encode', pos_encode)
            #since we would be concatenating
            # in_dim = in_dim + d
            self.proj_PE = nn.Linear(d,hidden_size,bias=False) #Transform the PE
            self.proj_input = nn.Linear(in_dim,hidden_size,bias=False)
        else:
            self.pos_encode = pos_encode

        self.spatial = ImpConGCN(hidden_size, hidden_size, use_bn=True)

        num_nodes = N # Just so I do not change variable name
        self.temporal = Transformer(hidden_size=num_nodes*hidden_size,filter_size=num_nodes*hidden_size)
        self.batch_norm = nn.BatchNorm2d(hidden_size)
        self.dropout_conv = nn.Dropout(0.1)

        # second layer
        # self.convert = nn.Linear(in_dim,hidden_size) # Result is poor without skip connection
        # self.spatial2 = ImpConGCN(hidden_size, hidden_size, use_bn=True)
        # self.temporal2 = Transformer(hidden_size=num_nodes*hidden_size,filter_size=num_nodes*hidden_size)
        # self.batch_norm2 = nn.BatchNorm2d(hidden_size)
        # self.dropout_conv2 = nn.Dropout(0.1)

        q = 1 #q decides how much we pool the final layer
        self.pool_result = nn.Linear(num_nodes, q) #for pooling the final layer

#######BATCH NORM NOT USED. FIND A WAY TO USE IT
        self.pool_batchnorm = nn.BatchNorm1d(q*hidden_size*time_dim)
        self.fc = MLP(num_layers=1, input_dim=q*hidden_size*time_dim, hidden_dim=1, output_dim=num_classes)

        initialize_weight(self.pool_result)


    def get_position_encoding(self, N, d):
        """This would return the saved position encoding"""
        PE = Position_encode(N=N, d=d)
        PE.load_state_dict(torch.load(os.getcwd()+'/DHCS_implement/Saved_models/'+f'd={d}checkpoint_pos_encode.pt'))
        PE.eval()
        pos_encode,_,_ = PE(test=True)
        # err = 0.0001
        # pos_encode = pos_encode + err
        # pos_encode = torch.round(pos_encode) # may remove approximation
        # Experiment shows that rounding (to force 0's & 1's) makes the result better
        return pos_encode


    def forward(self, features, A, reached_starting_epoch=False):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        out = features
        if self.pos_encode is not None:
            # b, t, _, _ = features.shape
            # we use .detach() for the position encoding
            # so it is not part of the computation graph when computing gradients.
            pos_encode = self.pos_encode.detach()#.unsqueeze(0).unsqueeze(0) #[1 x 1 x n_nodes x d]
            # pos_encode = pos_encode.repeat(b, t, 1, 1).to(features.device) #[b x time x n_nodes x d]
            # out = torch.cat([out, pos_encode], dim=-1) #[b x time x n_nodes x (d+in_dim)]
            out = self.proj_input(out) + self.proj_PE(pos_encode) #Add position encoding

        orig = out
        out = self.spatial(out , A) #[b x time x n_nodes x filter_size]
        out = self.temporal(out) #[b x time x n_nodes x filter_size]

        # skip connection
        # orig = self.convert(orig) + out
        orig = orig + out #[b x time x n_nodes x filter_size]
        # Batchnorm can be here
        orig = self.batch_norm(orig.transpose(-1,-3)).transpose(-1,-3)
        out = orig

        out = F.relu(out)
        out = self.dropout_conv(out) #dropout #[b x time x n_nodes x filter_size]

        #layer 2
        # out = self.spatial2(out , A) #[b x time x n_nodes x filter_size]
        # out = self.temporal2(out)

        # skip connection
        # orig = out + orig
        # # Batchnorm can be here
        # orig = self.batch_norm2(orig.transpose(-1,-3)).transpose(-1,-3)
        # out = orig
        #
        # out = F.relu(out)
        # out = self.dropout_conv2(out) #dropout #[b x time x n_nodes x filter_size]


        out = self.pool_result(out.transpose(2,3)).squeeze(-1) #[b x time x filter_size x q] #pooling
        out = out.flatten(1,-1) #[b x q*time*filter_size]
        out = self.pool_batchnorm(out)
        out = F.relu(out)

        return self.fc(out) #[b x num_class]
