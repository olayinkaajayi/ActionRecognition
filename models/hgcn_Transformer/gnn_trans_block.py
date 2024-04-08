import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from transformer import Transformer

class GNN_Trans(nn.Module):
    """
        This class would combine the GCN and LSTM modules to form our model for the video data.
    """

    def __init__(self, gnn, hidden_size, filter_size,pool=False):
        super(GNN_Trans, self).__init__()
        self.gnn = gnn
        self.batch_norm = nn.BatchNorm2d(hidden_size)
        self.transformer = Transformer(hidden_size=hidden_size,filter_size=filter_size,pool=pool)


    def forward(self,features,A):
        """
            data:   --dim(batch_size,time,num_nodes,in_channel)
            A:      --dim(num_nodes,num_nodes)
        """
        collector = []
        for j in range(features.size(1)): #loop across time dimension
            take = self.gnn(features[:,j],A)
            collector.append(take.unsqueeze(1)) #.unsqueeze(1) is to make it batch_size x 1 x num_nodes, to be stacked later across time dimension

        # if features.device.type == 'cuda':
        #     out = torch.cat(collector,1).cuda()
        # else: #shape is batch_size x time x num_nodes x out_channel
        out = torch.cat(collector,1).to(features.device)
        #take the result of the last step and pass to the transformer module

        #reshape for batch_norm
        # out = out.transpose(2,3) #[b x time x out_channel x n_nodes]
        # out = out.transpose(1,2) #[b x out_channel x time x n_nodes]
        # out = self.batch_norm(out)
        # #undo reshape
        # out = out.transpose(1,2) #[b x time x out_channel x n_nodes]
        # out = out.transpose(2,3) #[b x time x n_nodes x out_channel]
        #reshape done
        # out = F.relu(out) # The HGCN function already has a relu function

        out = self.transformer(out) #shape: [b x time x n_nodes x filter_size] if pool else [b x time x filter_size]
        #return final result to the calling model.
        return out
