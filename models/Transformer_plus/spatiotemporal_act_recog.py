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

    def __init__(self, mlp_numbers, hidden_size, gnn_model, filter_size=None, num_trans_layers=2,parts_gnn=False):
        super(Spatiotemp_Action_recog, self).__init__()

        time_dim, num_classes = mlp_numbers
        self.in_dim = gnn_model.dataset_num_features
        self.gnn_model = InfoGraph(gnn_model.dataset_num_features,gnn_model.hidden_dim,gnn_model.num_gc_layers)
        with torch.no_grad():
            for i in range(gnn_model.num_gc_layers):
                self.gnn_model.encoder.convs[i].conv.weight.copy_(gnn_model.encoder.convs[i].lin.weight)
                self.gnn_model.encoder.convs[i].conv.weight.requires_grad = False

        if filter_size is None:
            filter_size = hidden_size

        self.transformer = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()
        self.dropout_conv = torch.nn.ModuleList()
        self.convert1 = nn.Linear(self.in_dim,hidden_size)

        num_nodes = 25
        self.parts_gnn = parts_gnn
        if self.parts_gnn:
            self.pool_result = nn.Linear(num_nodes, 1) #for pooling the final layer

        self.num_trans_layers = num_trans_layers
        for i in range(num_trans_layers):
            if parts_gnn:
                self.transformer.append(Transformer_parts(hidden_size=num_nodes*hidden_size,filter_size=num_nodes*hidden_size))
                self.batch_norm.append(nn.BatchNorm2d(hidden_size) if i!= (num_trans_layers-1) else nn.BatchNorm1d(hidden_size*time_dim))
            else:
                self.transformer.append(Transformer(hidden_size=hidden_size,filter_size=filter_size))
                self.batch_norm.append(nn.BatchNorm1d(hidden_size))
            self.dropout_conv.append(nn.Dropout(0.1))

        # consider increasing num_layers to at least 2 (even in the other models)
        if self.parts_gnn:
            self.fc = MLP(num_layers=1, input_dim=hidden_size*time_dim, hidden_dim=1, output_dim=num_classes)
        else:
            self.fc = MLP(num_layers=3, input_dim=hidden_size*time_dim, hidden_dim=128, output_dim=num_classes)


    def A_tilde(self,A):
        """
            This function computes the degree matrix (D) from the adjacency matrix A.
            The return D^(-0.5) x A x D^(-0.5) as A_hat.
        """
        deg_mat = torch.sum(A, dim=1) # dimension is num_nodes
        tol = 1e-4 #tolerance to avoid taking the inverse of zero.
        frac_degree = torch.diag((deg_mat+tol)**(-0.5)).to(A.device)
        return torch.matmul(frac_degree,
                            torch.matmul(A,frac_degree)).float()


    def forward(self, features, A):
        """if pose-gnn:
            features:   --dim(batch_size,time,in_channel)
           if parts-gnn:
            features:   --dim(batch_size,time, num_nodes,in_channel)
        """
        A = A + np.eye(A.shape[0]) #Add an identity to A
        A = torch.from_numpy(A).to(features.device)
        A_hat = self.A_tilde(A)


        out = features
        out_l = features
        # Layers
        for i in range(self.num_trans_layers):

            out = F.relu(self.gnn_model.encoder.convs[i](out, A_hat))

            out = self.transformer[i](out) #[b x time x filter_size]
            try:
                out_l = out + out_l # Skip connection
            except RuntimeError:
                out_l = out + self.convert1(out_l)

            # batch_norm
            if self.parts_gnn and i!=(self.num_trans_layers-1):
                #reshape for batch_norm
                out = out_l.transpose(2,3) #[b x time x filter_size x n_nodes]
                out = out.transpose(1,2) #[b x filter_size x time x n_nodes]
                out = self.batch_norm[i](out)
                # undo reshape
                out = out.transpose(1,2) #[b x time x filter_size x n_nodes]
                out = out.transpose(2,3) #[b x time x n_nodes x filter_size]
                out = F.relu(out)
                out = self.dropout_conv[i](out)
                continue
                # reshape done
            if self.parts_gnn:
                out = self.pool_result(out_l.transpose(2,3)).squeeze(-1) #[b x time x filter_size] #pooling
                out = out.flatten(1,2) #[b x time*filter_size]
                out = self.batch_norm[i](out)
            else:
                out = self.batch_norm[i](out_l.transpose(1,2)).transpose(1,2)
            out = F.relu(out)
            out = self.dropout_conv[i](out) #dropout #[b x time x filter_size]

        if not self.parts_gnn:
            out = out.flatten(1,2) #[b x time*filter_size]

        return self.fc(out)
