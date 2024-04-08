import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Local_con_net(nn.Module):
    """This is a class that helps initialize the weights for the model"""

    def __init__(self, num_nodes, in_dim, out_dim=0, batch_Norm=False):
        super(Local_con_net, self).__init__()
        self.in_dim = in_dim
        if out_dim == 0: #We do not want the output dimension to change from the input_dim
            self.out_dim = in_dim
        else:
            self.out_dim = out_dim

        self.batch_Norm = batch_Norm
        if self.batch_Norm:
            self.normalize = nn.BatchNorm1d(self.out_dim)

        self.num_nodes = num_nodes
        #Dimension is num_nodes, num_nodes, in_dim, out_dim to account for the embedding of each node
        #And the relationship each node has with neighbouring node
        self.weight = nn.Parameter(
                torch.FloatTensor(num_nodes, num_nodes, in_dim, self.out_dim))
        #Find out how to add bias
        #self.bias = nn.Parameter(torch.FloatTensor(out_dim))

        #This weight would help "pool" the transformed features
        self.pool_weight = nn.Parameter(torch.FloatTensor(self.out_dim,1))

        init.xavier_uniform_(self.weight) # I think this initialization are affecting
        init.xavier_uniform_(self.pool_weight)
        #init.constant_(self.bias, 0)

    def __pooling(self,input):
        """
            This returns the pooled result of the input
            input: --dim(batch_size,num_nodes,out_dim)
        """
        #You can consider using nn.Linear() for this.
        batch_size = input.size(0)
        return torch.bmm(input,self.pool_weight.unsqueeze(0).repeat(batch_size,1,1)).squeeze(1) #.squeeze(1) emsures the output dimension is batch_size x num_nodes


    def forward(self,features,A,pool=True):
        """
            features:   feature matrix--dim(batch_size,num_nodes,in_channel)
            A:          adjacency matrix--dim(num_nodes,num_nodes)
        """
        batch_size = features.size(0)
        #This is a matrix whose rows contains h_i across each batch
        if features.device.type == 'cuda':
            H = torch.FloatTensor(batch_size,self.num_nodes,1,self.out_dim).cuda()
        else: #The dimension 1 is just to allow it accomodate the returned array of dummy_var
            H = torch.FloatTensor(batch_size,self.num_nodes,1,self.out_dim).to(features.device)

        for i in range(self.num_nodes):
            dummy_var = 0
            for j in range(self.num_nodes):
                #.unsqueeze(1) is needed to keep the Dimension as [batch_size,1,in_dim]
                dummy_var += A[i,j]*torch.bmm(features[:,j].unsqueeze(1),self.weight[i,j,:].unsqueeze(0).repeat(batch_size,1,1))
            H[:,i] = dummy_var
        H = H.squeeze(-2) #This is to undo the unnecessary dimension 1 added above.
        if self.batch_Norm:
            H = self.normalize(H.transpose(1,2)).transpose(1,2) #batch normalization requires data of shape batch_size x in_channel x num_nodes
        #consider putting a ReLU on H ie. F.ReLU(H) to give non-linearity
        H = F.relu(H)
        if pool: #This would be useful when implementing the Dynamic Hierarchical Channel Squeezing module (pool will set to False)
            return self.__pooling(H)
        return H
