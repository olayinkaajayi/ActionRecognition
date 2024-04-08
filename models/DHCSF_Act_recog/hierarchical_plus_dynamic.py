import torch
import torch.nn as nn

import sys
# sys.path.append("DHCS_implement/models")
from dynamic_graph import Dynamic_graph
from hierarchical_lcn import Hier_LCN

class DHCSF_p_TCN(nn.Module):
    """This class is meant to implement the Hierarchical Channel-Squeezing Fusion
        layer (D-HCSF) and the Temporal Connected Network (TCN) ie. dynamic graph as
        discussed in the paper: Learning Skeletal Graph Neural Networks for
        Hard 3D Pose Estimation.
    """
    def __init__(self,A,S,L,d,F,num_nodes,in_dim,out_dim_HCSF,conv_out_dim):
        super(DHCSF_p_TCN, self).__init__()
        self.DHCSF = Hier_LCN(S,L,d,A,num_nodes,in_dim,out_dim_HCSF)
        self.TCN = Dynamic_graph(num_nodes, in_dim, conv_out_dim, F)

    def forward(self,features,A_k):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A_k:        --dim(num_nodes,num_nodes) this is the result of A + alpha*O_k
        """
        collector = []
        for i in range(features.size(1)): #loop across time dimension
            out = self.DHCSF(features[:,i],A_k)
            collector.append(out.unsqueeze(1)) #.unsqueeze(1) is to make it batch_size x 1 x num_nodes, to be stacked later across time dimension

        if features.device.type == 'cuda':
            collector = torch.cat(collector,1).cuda()
        else: #shape is batch_size x time x num_nodes x in_channel (used by D-HCSF again)
            collector = torch.cat(collector,1).to(features.device)

        out_O_k = self.TCN(collector)

        return collector , out_O_k
