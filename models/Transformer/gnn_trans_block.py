import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

# from transformer import Transformer
from transformer_no_pool import Transformer
# from gcn_model import GCN

class GNN_Trans(nn.Module):
    """
        This class would combine the GCN and LSTM modules to form our model for the video data.
    """

    def __init__(self, gnn, hidden_size, filter_size,pool=False):
        super(GNN_Trans, self).__init__()
        self.gnn = gnn
        # self.batch_norm = nn.BatchNorm2d(hidden_size)
        self.transformer = Transformer(hidden_size=hidden_size,filter_size=filter_size,pool=pool)


    def forward(self,features,A):
        """
            data:   --dim(batch_size,time,num_nodes,in_channel)
            A:      --dim(batch_size,num_nodes,num_nodes)
        """
        out = self.gnn(features,A) #[b x time x n_nodes x out_channel]

        #reshape for batch_norm
#         out = out.transpose(2,3) #[b x time x out_channel x n_nodes]
#         out = out.transpose(1,2) #[b x out_channel x time x n_nodes]
# ########THIS BATCHNORM IS CAUSING THE PROBLEM
#         out = self.batch_norm(out)
#         #undo reshape
#         out = out.transpose(1,2) #[b x time x out_channel x n_nodes]
#         out = out.transpose(2,3) #[b x time x n_nodes x out_channel]
#         #reshape done
        out = F.relu(out)

        out = self.transformer(out) #shape: [b x time x n_nodes x filter_size] if pool else [b x time x filter_size]
        #return final result to the calling model.
        return out
