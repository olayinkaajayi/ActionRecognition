import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from gcn_model import GCN

class GCN_stack(nn.Module):
    """This module implements a stacked GCN."""
    def __init__(self, in_dim, out_dim, num_stacks=3, average=False):
        super(GCN_stack, self).__init__()
        self.num_stacks = num_stacks
        self.average = average #whether average pooling or learned pooling
        self.gcn = nn.ModuleList()

        self.gcn.append( GCN(in_dim,out_dim) )
        #initialization
        use_first_n_last = False
        if out_dim != 0:
            use_first_n_last = True
            first = in_dim
            last = out_dim
        for i in range(1,num_stacks):
            if (out_dim == 0) and (not use_first_n_last):
                stack_in_dim = (2**i)*in_dim
                if i == (num_stacks-1): #to set dimension of learned weight
                    final = 2*stack_in_dim
            else:
                stack_in_dim = self.first_n_last(first,last)
                first = stack_in_dim
                last = stack_in_dim
                out_dim = 0 #so that out_dim_GCN = stack_in_dim
                if i == (num_stacks-1): #to set dimension of learned weight
                    final = self.first_n_last(first,last)
            self.gcn.append(GCN(stack_in_dim,out_dim))

        if not average:
            print("Learned pooling used")
            self.pool_weight = nn.Parameter(torch.FloatTensor(final,1)) #This weight would help "pool" the transformed features
            init.xavier_uniform_(self.pool_weight)

    def first_n_last(self,a,b):
        """This function is meant to just sum"""
        return a + b

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
            return torch.sum(input,-2) #changed from -1 to -2


    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        out = features
        for i in range(self.num_stacks):
            tmp = out
            ####Do across time domain
            collector = []
            for j in range(out.size(1)): #loop across time dimension
                take = self.gcn[i](out[:,j],A)
                collector.append(take.unsqueeze(1)) #.unsqueeze(1) is to make it batch_size x 1 x num_nodes, to be stacked later across time dimension

            if features.device.type == 'cuda':
                out = torch.cat(collector,1).cuda()
            else: #shape is batch_size x time x num_nodes x in_channel
                out = torch.cat(collector,1).to(features.device)

            out = torch.cat((out,tmp),-1)
        return self.pooling(out,self.average)
