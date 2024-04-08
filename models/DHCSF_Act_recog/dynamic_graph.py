import torch
import torch.nn as nn

class Dynamic_graph(nn.Module):
    """This class implements the dynamic graph module discussed in the paper: Learning Skeletal Graph Neural
       Networks for Hard 3D Pose Estimation.
    """

    def __init__(self, num_nodes, in_dim, conv_out_dim, F):
        super(Dynamic_graph, self).__init__()
        #F is kernel size
        self.W_theta = nn.ModuleList([nn.Conv1d(in_dim,conv_out_dim,F) for i in range(num_nodes)])
        self.W_phi = nn.ModuleList([nn.Conv1d(in_dim,conv_out_dim,F) for i in range(num_nodes)])
        # self.alpha = nn.Parameter(torch.FloatTensor(1))
        self.num_nodes = num_nodes
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(self.num_nodes)


    def forward(self,features):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
        """
        data = features.transpose(1,2) #now has shape batch_size x num_nodes x time x in_channel
        data = data.transpose(2,3) #now has shape batch_size x num_nodes x in_channel x time
        W_theta_holder = []
        W_phi_holder = []
        #start with 1D convolution
        for i in range(self.num_nodes):
            #flatten the result across the time domain to get batch_size x out_dim*time_out
            W_theta_holder.append(self.W_theta[i](data[:,i]).flatten(start_dim=1).unsqueeze(1)) #.unsqueeze(1) is to set it up for stacking up later
            W_phi_holder.append(self.W_phi[i](data[:,i]).flatten(start_dim=1).unsqueeze(-1)) #.unsqueeze(-1) is to set it up for stacking up later

        #make dimension batch_size x num_nodes x out_dim*time_out
        W_theta_result = torch.cat(W_theta_holder,1)
        #make dimension batch_size x out_dim*time_out x num_nodes
        W_phi_result = torch.cat(W_phi_holder,-1)
        #Then multiply both result of transformation together to get num_nodes x num_nodes
        tmp = torch.bmm(W_theta_result,W_phi_result)
        #pass though Tanh to get the O_k weight matrix.
        O_k = self.tanh(tmp)
        #Put ReLU and BatchNorm1d here
        O_k = self.batch_norm(O_k)
        O_k = self.relu(O_k)

        return O_k
