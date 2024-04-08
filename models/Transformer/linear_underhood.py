import torch.nn as nn
from torch.nn import Linear

class MyLinear(nn.Module):
    """
        This class is for a linear transformation.
    """

    def __init__(self, in_dim,out_dim):
        super(MyLinear, self).__init__()
        self.linear = Linear(in_dim,out_dim)


    def forward(self,features,A):
        """
            data:   --dim(batch_size,time,num_nodes,in_channel)
            A:      --dim(batch_size,num_nodes,num_nodes)
        """
        return self.linear(features) #[b x time x n_nodes x out_channel]
