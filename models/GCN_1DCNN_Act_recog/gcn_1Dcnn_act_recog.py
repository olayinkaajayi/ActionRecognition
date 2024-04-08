import numpy as np
import torch
import torch.nn as nn

from gcn_1Dcnn_block import GCN_1DCNN
from mlp import MLP

class GCN_1dcnn_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, mlp_numbers, cnn_numbers, num_nodes, in_dim, out_dim):
        super(GCN_1dcnn_Action_recog, self).__init__()

        cnn_out_dim, kernel_len, cnn_stride = cnn_numbers
        time_dim , num_layers , hidden_dim , output_dim = mlp_numbers
        L_out_conv = int((time_dim - (kernel_len - 1) - 1)/cnn_stride + 1)

        pool_stride = 4
        pool_kernel_len = kernel_len//4
        L_out_pool = int((L_out_conv - pool_kernel_len)/pool_stride + 1)

        self.gcn_cnn = GCN_1DCNN( num_nodes, cnn_numbers, in_dim, out_dim)

        if out_dim == 0:
            out_dim = in_dim
        if cnn_out_dim == 0:
            cnn_out_dim = out_dim

        self.fc = MLP(num_layers=num_layers, input_dim=(in_dim+cnn_out_dim)*L_out_pool, hidden_dim=hidden_dim, output_dim=output_dim)


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

        out = self.gcn_cnn(features , A_hat) # batch_size x channel_out x pool_len_out
        out = out.flatten(1,2)
        return self.fc(out)
