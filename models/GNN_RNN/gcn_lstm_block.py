import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from full_LSTM_module import Lstm_block
from gcn_model import GCN

class GCN_LSTM(nn.Module):
    """
        This class would combine the GCN and LSTM modules to form our model for the video data.
    """

    def __init__(self, num_nodes, in_dim, out_dim=0, hidden_dim=0):
        super(GCN_LSTM, self).__init__()
        self.gcn = GCN(in_dim, out_dim)
        self.lstm = Lstm_block(input_dim=num_nodes,hidden_dim=hidden_dim)


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
        #take the result of the last step and pass to the lstm module
        out = self.lstm(out.sum(dim=-1)) #sum is for pooling #shape: batch_size x hidden_dim
        #return final result to the calling model.
        return out
