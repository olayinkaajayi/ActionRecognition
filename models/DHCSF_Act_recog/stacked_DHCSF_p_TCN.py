import torch
import torch.nn as nn

from hierarchical_plus_dynamic import DHCSF_p_TCN

class Stack_DHCSF(nn.Module):
    """This module implements a stacked D-HCSF layer + TCN."""
    def __init__(self, A,S,L,d,F,num_nodes,in_dim,out_dim_HCSF,conv_out_dim,num_stacks=3,average=False):
        super(Stack_DHCSF, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1))
        self.num_stacks = num_stacks
        self.average = average #whether average pooling or learned pooling
        self.dh_p_tcn = nn.ModuleList()

        self.dh_p_tcn.append( DHCSF_p_TCN(A,S,L,d,F,num_nodes,in_dim,out_dim_HCSF,conv_out_dim))
        for i in range(1,num_stacks): #changed
            if out_dim_HCSF == 0:
                stack_in_dim = (2**(i-0))*in_dim #changed
            else:
                holder = out_dim_HCSF if i == 1 else holder #changed
                stack_in_dim = (2**(i-0))*holder #This is due to the skip connections #changed
                out_dim_HCSF = 0 #so that out_dim_HCSF = stack_in_dim
            self.dh_p_tcn.append(DHCSF_p_TCN(A,S,L,d,F,num_nodes,stack_in_dim,out_dim_HCSF,conv_out_dim))
        # 2*stack_in_dim is the output dimension for the last layer
        if not average:
            self.pool_weight = nn.Parameter(torch.FloatTensor(2*stack_in_dim,1)) #This weight would help "pool" the transformed features
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
            return torch.mean(input,-1)


    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        # We need A across the batches
        if features.device.type == 'cuda':
            A_k = torch.FloatTensor(A).repeat(features.size(0),1,1).cuda() # A_k is the result of A + alpha*O_k
            A = torch.FloatTensor(A).repeat(features.size(0),1,1).cuda()
        else:
            A_k = torch.FloatTensor(A).repeat(features.size(0),1,1).to(features.device)
            A = torch.FloatTensor(A).repeat(features.size(0),1,1).to(features.device)

        out = features
        for i in range(self.num_stacks):
            tmp = out
            out,O_k = self.dh_p_tcn[i](out,A_k)
            A_k = A + self.alpha*O_k
            if i>=0 : #changed
                out = torch.cat((out,tmp),-1)
        return self.pooling(out,self.average)
