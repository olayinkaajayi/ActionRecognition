import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from torch.nn import init

from trans_pos_encode import TT_Pos_Encode
from multi_stream import MultiStream
from mlp import MLP
from final_interact import InteractFinal

class ActionRecognition(nn.Module):
    """This module couples the ActionRecognition model together using multiple multisteam layers."""

    def __init__(self, in_dim, hidden_size, mlp_numbers, num_nodes=25, d=8,
                pred_concat=False, PE_name='', use_PE=True, use_intr=True):
        super(ActionRecognition, self).__init__()

        time_dim, num_classes = mlp_numbers
        self.N = num_nodes
        self.hidden_size = hidden_size
        # Position Encoding
        if use_PE:
            PE = TT_Pos_Encode(hidden_size, num_nodes, d, PE_name)
            pos_encode = PE.get_position_encoding(need_plot=False)
            self.register_buffer('pos_encode', pos_encode)
            # print(f'RG:{self.pos_encode.requires_grad}')
        else:
            print("\n\n!!!NOT USING PE!!!\n\n")
            self.pos_encode = None
        self.proj_input = nn.Linear(in_dim,hidden_size,bias=False)
        init.xavier_uniform_(self.proj_input.weight)

        # Multistream modules
        self.multistream1 = MultiStream(hidden_size, num_nodes, layer1=True, use_intr=use_intr)
        self.multistream2 = MultiStream(hidden_size, num_nodes, use_intr=use_intr)

        # Pooling layer (we pool the graph into 1 vector embedding)
        self.pool_x1 = nn.Linear(num_nodes, 1) # for pooling the final layer for the skeleton
        self.pool_x2 = nn.Linear(num_nodes, 1) # for pooling the final layer for the skeleton

        init.xavier_uniform_(self.pool_x1.weight)
        init.xavier_uniform_(self.pool_x2.weight)

        # Normalization for pooling
        self.pool_batchnorm_x1 = nn.BatchNorm1d(hidden_size*time_dim)
        self.pool_batchnorm_x2 = nn.BatchNorm1d(hidden_size*time_dim)

        # Get mean and variance for noise
        q = 2 #For the 2 final streams
        # self.noise_ratio = 0.1
        # Change this to 64 each
        # self.out_dim = 128# hidden_size*2 #each for mean and std
        # self.mean_n_std = MLP(num_layers=2, input_dim=q*hidden_size*time_dim, hidden_dim=hidden_size, output_dim=(self.out_dim//2))

        # Prediction layer
        # self.fc = MLP(num_layers=3, input_dim=(self.out_dim//2), hidden_dim=self.out_dim, output_dim=num_classes)
        self.pred_concat = pred_concat # Determines what we do with the final layer
        if pred_concat:
            self.fc = MLP(num_layers=1, input_dim=q*hidden_size*time_dim, hidden_dim=1, output_dim=num_classes)
        else:
            self.fc = nn.ModuleList()
            for i in range(q):
                self.fc.append(MLP(num_layers=1, input_dim=hidden_size*time_dim, hidden_dim=1, output_dim=num_classes))


    def add_noise(self, mu, logvar):
        """This is where we arr the Gaussian noise to the encoding"""
        if self.training:
            std = logvar.mul(self.noise_ratio).exp()
            # std = logvar.exp()
            std = torch.clamp(std, max=100)
            # std = std / (torch.norm(std, 2, dim=1, keepdim=True) + 1e-4)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std) + mu
        else:
            return mu


    def forward(self, features, A, epoch=None):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """

        features = self.proj_input(features)
        if self.pos_encode is not None:
            # we use .detach() for the position encoding
            # so it is not part of the computation graph when computing gradients.
            pos_encode = self.pos_encode.detach()
            features = features + pos_encode.repeat(2,1) #Add position encoding

        x1, x2 = features[:,:,:self.N], features[:,:,self.N:]

        # First layer
        x1, x2 = self.multistream1(x1, x2, A)
        x1, x2 = F.relu(x1), F.relu(x2)

        # Second layer
        x1, x2 = self.multistream2(x1, x2, A)
        x1, x2 = F.relu(x1), F.relu(x2)

        # Pool and normalize
        x1 = self.pool_x1(x1.transpose(2,3)).squeeze(-1) #[b x time x filter_size]

        x2 = self.pool_x2(x2.transpose(2,3)).squeeze(-1) #[b x time x filter_size]

        x1 = x1.flatten(1,-1) #[b x time*filter_size]
        x1 = self.pool_batchnorm_x1(x1)

        x2 = x2.flatten(1,-1) #[b x time*filter_size]
        x2 = self.pool_batchnorm_x2(x2)

        # ReLu
        x1, x2 = F.relu(x1), F.relu(x2)

        # Mean n std
        # x = self.mean_n_std(torch.concat([x1,x2], dim=-1))
        # z = x #self.add_noise(x[:,:64],x[:,64:])

        # Prediction scores
        # scores = self.fc(z)
        if self.pred_concat:
            return self.fc(torch.concat([x1,x2], dim=-1))
        else: # We sum the individual scores
            out = 0
            outputs = [x1, x2] #[x1+x2]
            for i,each in enumerate(outputs):
                out += self.fc[i](each)

        return out# scores , z
