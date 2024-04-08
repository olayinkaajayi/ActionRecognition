import torch
import torch.nn as nn
import numpy as np

from gcn_stacked import GCN_stack
from mlp import MLP

class GCN_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, mlp_numbers, num_nodes, in_dim, out_dim, num_stacks=3, average=True):
        super(GCN_Action_recog, self).__init__()
        self.stacks = GCN_stack(in_dim, out_dim, num_stacks, average)
        time_dim , num_layers , hidden_dim , output_dim = mlp_numbers
        # input_dim is p*time_dim where p is dependent on the output of stacks
        self.fc = MLP(num_layers=num_layers, input_dim=24*time_dim, hidden_dim=hidden_dim, output_dim=output_dim)


    def A_tilde(self,A):
        """
            This function computes the degree matrix (D) from the adjacency matrix A.
            The return D^(-0.5) x A x D^(-0.5) as A_hat.
        """
        deg_mat = torch.sum(A, dim=1) # dimension is num_nodes
        tol = 1e-4 #tolerance to avoid taking the inverse of zero.
        if A.device.type == 'cuda':
            frac_degree = torch.diag((deg_mat+tol)**(-0.5)).cuda() # The power of a diagonal matrix is the power of it's entries
        else:
            frac_degree = torch.diag((deg_mat+tol)**(-0.5)).to(A.device)
        return torch.matmul(frac_degree,
                            torch.matmul(A,frac_degree)).float()


    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        A = A + np.eye(A.shape[0]) #Add an identity to A
        if features.device.type == 'cuda':
            A = torch.from_numpy(A).cuda()
        else:
            A = torch.from_numpy(A).to(features.device)
        A_hat = self.A_tilde(A)
        out = self.stacks(features , A_hat)
        out = out.flatten(1) #batch_size x num_nodes*time_dim

        return self.fc(out)
