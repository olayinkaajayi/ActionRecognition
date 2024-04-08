import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gcn_model import GCN

class GCN_1DCNN(nn.Module):
    """
        This class would combine the GCN and LSTM modules to form our model for the video data.
    """

    def __init__(self, num_nodes, cnn_numbers, in_dim, out_dim=0, pool_stride=4):
        super(GCN_1DCNN, self).__init__()
        self.gcn = GCN(in_dim, out_dim)
        if out_dim == 0:
            out_dim = in_dim
        cnn_out_dim, kernel_len, cnn_stride = cnn_numbers
        if cnn_out_dim == 0:
            cnn_out_dim = out_dim
        self.temporal = nn.Sequential(nn.Conv1d(in_dim+out_dim,in_dim+cnn_out_dim,kernel_len,stride=cnn_stride),
                                      nn.ReLU()
                                      )
        pool_kernel_len = kernel_len//4
        self.pool = nn.AvgPool1d(pool_kernel_len,stride=pool_stride)

    def forward(self,features,A):
        """
            data:   --dim(batch_size,time,num_nodes,in_channel)
            A:      --dim(num_nodes,num_nodes)
        """
        collector = []
        for j in range(features.size(1)): #loop across time dimension
            take = self.gcn(features[:,j],A)
            collector.append(take.unsqueeze(1)) #.unsqueeze(1) is to make it batch_size x 1 x num_nodes, to be stacked later across time dimension

        if features.device.type == 'cuda':
            out = torch.cat(collector,1).cuda()
        else: #shape is batch_size x time x num_nodes x in_channel (used by D-HCSF again)
            out = torch.cat(collector,1).to(features.device)

        out = torch.cat((out,features),-1)

        out = out.sum(dim=2) #pooling: batch_size x time x in_channel
        out = torch.transpose(out,1,2) #batch_size x in_channel x time_dim

        out = self.temporal(out) #batch_size x channel_out x time_len_out

        out = self.pool(out) # batch_size x channel_out x pool_len_out
        return out
