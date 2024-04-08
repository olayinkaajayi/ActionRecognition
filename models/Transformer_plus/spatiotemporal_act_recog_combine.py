import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mlp import MLP
from infograph_mod import InfoGraph
from transformer import Transformer
from Transformer.transformer_no_pool import Transformer as Transformer_parts
from transformer import initialize_weight

class Spatiotemp_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, in_dim, mlp_numbers, hidden_size, filter_size=None, num_gc_layers=2, num_trans_layers=2,parts_gnn=False,A=None):
        super(Spatiotemp_Action_recog, self).__init__()
        if A is not None:
            A = torch.tensor(A+np.eye(A.shape[0]),dtype=torch.float32,requires_grad=False)
            self.register_buffer('A', A)
        else:
            self.A = A
        time_dim, num_classes = mlp_numbers
        self.in_dim = in_dim
        self.hidden_size = hidden_size
        self.gnn_model = InfoGraph(self.in_dim,hidden_size,num_gc_layers)

        if filter_size is None:
            filter_size = hidden_size

        self.transformer = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout_conv = nn.ModuleList()
        # self.convert1 = nn.Linear(self.in_dim,hidden_size)

        num_nodes = 25
        self.parts_gnn = parts_gnn
        if self.parts_gnn:
            self.pool_result = nn.Linear(num_nodes, 1) #for pooling the final layer
            self.pool_batchnorm = nn.BatchNorm1d(hidden_size*time_dim)

        self.num_trans_layers = num_trans_layers
        for i in range(num_trans_layers):
            if parts_gnn:
                self.transformer.append(Transformer_parts(hidden_size=num_nodes*hidden_size,filter_size=num_nodes*hidden_size))
                self.batch_norm.append(nn.BatchNorm2d(hidden_size))
            else:
                self.transformer.append(Transformer(hidden_size=hidden_size,filter_size=filter_size))
                self.batch_norm.append(nn.BatchNorm1d(hidden_size))
            self.dropout_conv.append(nn.Dropout(0.1))

        # consider increasing num_layers to at least 2 (even in the other models)
        if self.parts_gnn:
            self.fc = MLP(num_layers=1, input_dim=hidden_size*time_dim, hidden_dim=1, output_dim=num_classes)
        else:
            self.fc = MLP(num_layers=3, input_dim=hidden_size*time_dim, hidden_dim=128, output_dim=num_classes)



    def forward(self, features, A):
        """if pose-gnn:
            features:   --dim(batch_size,time,in_channel)
           if parts-gnn:
            features:   --dim(batch_size,time, num_nodes,in_channel)
        """
        if self.A is not None:
            A = self.A

        # out = features
        # out_l = features
        g_enc, l_enc, neg_enc = self.gnn_model(features,A)
        out = l_enc[:,:,:,:self.hidden_size] #using just the output of the first GCN layer
        out_l = out
        # Layers
        for i in range(self.num_trans_layers):
            if i>0:
                out = F.relu(self.gnn_model.encoder.convs[i](out, A))

            out = self.transformer[i](out) #[b x time x filter_size]
            # try:
            out_l = out + out_l # Skip connection
            # except RuntimeError:
            #     out_l = out + self.convert1(out_l)

            # batch_norm
            if self.parts_gnn:# and i!=(self.num_trans_layers-1):
                #reshape for batch_norm
                out = out_l.transpose(2,3) #[b x time x filter_size x n_nodes]
                out = out.transpose(1,2) #[b x filter_size x time x n_nodes]
                out = self.batch_norm[i](out)
                # undo reshape
                out = out.transpose(1,2) #[b x time x filter_size x n_nodes]
                out = out.transpose(2,3) #[b x time x n_nodes x filter_size]
                out = F.relu(out)
                out = self.dropout_conv[i](out)
                # continue
                # reshape done
        if self.parts_gnn:
            out = self.pool_result(out.transpose(2,3)).squeeze(-1) #[b x time x filter_size] #pooling
            out = out.flatten(1,2) #[b x time*filter_size]
            out = self.pool_batchnorm(out)
            # out = F.relu(out)
        else:
            out = self.batch_norm[i](out_l.transpose(1,2)).transpose(1,2)
            out = F.relu(out)
            out = self.dropout_conv[i](out) #dropout #[b x time x filter_size]

        if not self.parts_gnn:
            out = out.flatten(1,2) #[b x time*filter_size]

        return self.fc(out), g_enc, l_enc, neg_enc
