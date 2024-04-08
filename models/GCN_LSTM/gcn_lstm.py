import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from full_LSTM_module import Lstm_block
from gcn_stacked import GCN_stack

class GCN_p_LSTM(nn.Module):
    """
        This class would combine the GCN and LSTM modules to form our model for the video data.
    """

    def __init__(self, num_class, num_nodes, in_dim, out_dim=0, hidden_dim=0, num_stacks=3, average=True):
        super(GCN_p_LSTM, self).__init__()
        self.stacks = GCN_stack(in_dim, out_dim, num_stacks, average)
        self.hidden_dim = hidden_dim
        self.lstm = Lstm_block(input_dim=num_nodes,output_dim=num_class,hidden_dim=self.hidden_dim)


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


    def forward(self,features,A):
        """
            data:   --dim(batch_size,time,num_nodes,in_channel)
            A:      --dim(num_nodes,num_nodes)
        """
        A = A + np.eye(A.shape[0]) #Add an identity to A
        if features.device.type == 'cuda':
            A = torch.from_numpy(A).cuda()
        else:
            A = torch.from_numpy(A).to(features.device)
        A_hat = self.A_tilde(A)
        out = self.stacks(features , A_hat) #dim is batch_size x time x num_nodes
        
        #take the result of the last step and pass to the lstm module
        out = self.lstm(out)
        #return final result to the calling model.
        return out
