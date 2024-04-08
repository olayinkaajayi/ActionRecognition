import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class GNN_for_frames(nn.Module):
    """This module implements a stacked GCN."""
    def __init__(self, time_len, num_nodes, gnn, out_dim, average=False):
        super(GNN_for_frames, self).__init__()
        self.time_len = time_len
        self.num_nodes = num_nodes
        self.average = average #whether average pooling or learned pooling
        self.gnn = gnn
        self.batch_norm = nn.BatchNorm1d(64)
        if not average:
            print("Learned pooling used")
            self.pool_weight = nn.Parameter(torch.FloatTensor(final,1)) #This weight would help "pool" the transformed features
            init.xavier_uniform_(self.pool_weight)

    def pooling(self,input,average=False):
        """
            This returns the pooled result of the input
            input: --dim(batch_size,time,num_nodes,out_dim)
        """
        #You can consider using nn.Linear() for this or just average
        if not self.average:
            batch_size = input.size(0)
            time_len = input.size(1)
            #.squeeze(-1) ensures the output dimension is batch_size x num_nodes
            #.unsqueeze(1) ensures the output dimension is batch_size x 1 x num_nodes (needed for stacking later)
            holder = [torch.bmm(input[:,i],self.pool_weight.unsqueeze(0).repeat(batch_size,1,1)).squeeze(-1).unsqueeze(1) for i in range(time_len)]
            return torch.cat(holder,1)
        else:
            #This has now been edited
            return torch.sum(input,-1)


    def forward(self, data):
        """
            data is of type torch_geometric.data.dataset
        """
        batch_size = data.y.size(0)
        out = self.gnn(data) #[b x out_channel]
        out = self.batch_norm(out) #batch_norm
        out = out.reshape(batch_size,self.time_len,self.num_nodes,-1) #.unsqueeze(1) is to make it batch_size x 1 x num_nodes, to be stacked later across time dimension

        return self.pooling(out,self.average)
