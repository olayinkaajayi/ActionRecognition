import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class GCN(nn.Module):
    """This module implements the vanilla GCN model."""

    def __init__(self, in_dim, out_dim, use_bn=False, layer1=None):
        super(GCN, self).__init__()

        if out_dim == 0:
            out_dim = in_dim

        self.conv = nn.Linear(in_dim,out_dim,bias=False)
        init.xavier_uniform_(self.conv.weight)

        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_dim)


    def A_tilde(self, A):
        """
            This function computes the degree matrix (D) from the adjacency matrix A.
            The return D^(-0.5) x A x D^(-0.5) as A_hat.
        """
        if len(A.shape) <= 2:
            deg_mat = torch.sum(A, dim=-1) # dimension is num_nodes
            tol = 1e-4 #tolerance to avoid taking the inverse of zero.
            frac_degree = torch.diag((deg_mat+tol)**(-0.5)).to(A.device)
            return torch.matmul(frac_degree,
                                torch.matmul(A,frac_degree)).float()

        else:
            deg_mat = torch.sum(A, dim=-1) # dimension is num_nodes
            tol = 1e-4 #tolerance to avoid taking the inverse of zero.
            b, t, n = deg_mat.shape
            ######################################################
            # Find a way to make this faster
            diags = torch.zeros(b,t,n,n)
            for i in range(b):
                for j in range(t):
                    diags[i,j,:,:] = torch.diag(deg_mat[i,j,:])
            frac_degree = ((diags+tol)**(-0.5)).to(A.device)
            ######################################################
            return torch.matmul(frac_degree,
                                torch.matmul(A,frac_degree)).float()


    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        if 'torch' in str(type(A)):
            A = A + torch.eye(A.shape[-1]) #Add an identity to A
        else:
            A = A + np.eye(A.shape[0]) #Add an identity to A
            A = torch.from_numpy(A).to(features.device)
        A = self.A_tilde(A)

        out = self.conv(features) #[b x time x n_nodes x out_channel]
        if len(A.shape) <= 2 :
            out = out.transpose(-1,-2).matmul(A) #[b x time x out_channel x n_nodes]
        else: #The second case is for when we have a matrix for each skeleton frame
            out = out.matmul(A).transpose(-1,-2) #[b x time x out_channel x n_nodes]

        if self.use_bn:
            out = self.bn(out.transpose(1,2)) #[b x out_channel x time x n_nodes]
            out = out.transpose(1,2)#[b x time x out_channel x n_nodes]

        return out.transpose(-1,-2) #[b x time x n_nodes x out_channel]
