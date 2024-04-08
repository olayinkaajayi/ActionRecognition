import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from infograph_mod import InfoGraph
from mlp import MLP
from Transformer_parts.transformer import initialize_weight
from Transformer_parts.transformer import Transformer as Transformer_body


class Body_GNN_Trans(torch.nn.Module):
    def __init__(self, in_dim, hidden_size, mlp_numbers, num_trans_layers=2, num_gc_layers=2):
        super(Body_GNN_Trans, self).__init__()

        time_dim, num_classes = mlp_numbers

        if num_trans_layers is None:
            num_trans_layers = num_gc_layers

        self.gnn_model = InfoGraph(in_dim, hidden_size, num_gc_layers)
        self.transformer = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.dropout_conv = nn.ModuleList()

        self.num_trans_layers = num_trans_layers
        filter_size = 16
        for i in range(num_trans_layers):
            self.transformer.append(Transformer_body(hidden_size=num_gc_layers*hidden_size,filter_size=filter_size))
            self.batch_norm.append(nn.BatchNorm1d(num_gc_layers*hidden_size))
            self.dropout_conv.append(nn.Dropout(0.1))

        self.fc = MLP(num_layers=1, input_dim=num_gc_layers*hidden_size*time_dim, hidden_dim=128, output_dim=num_classes)


    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """

        g_enc, l_enc, neg_enc = self.gnn_model(features,A)
        out = g_enc
        out_l = g_enc #used for skip connection
        # Layers
        for i in range(self.num_trans_layers):

            out = self.transformer[i](out) #[b x time x filter_size]

            out_l = out + out_l # Skip connection

            # batch_norm
            out = self.batch_norm[i](out_l.transpose(1,2)).transpose(1,2)
            out = F.relu(out)
            out = self.dropout_conv[i](out) #dropout #[b x time x filter_size]

        out = out.flatten(1,2) #[b x time*filter_size]

        return self.fc(out), g_enc, l_enc, neg_enc
