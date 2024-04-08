import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch.nn import init

class Hier_LCN(nn.Module):
    """This class builds the hierarchical locally connected network
        as described in the paper: Learning Skeletal Graph Neural
        Networks for Hard 3D Pose Estimation."""

    def __init__(self,S,L,d,A,num_nodes,in_dim,out_dim=0):
        super(Hier_LCN, self).__init__()
        self.S = S #short distance range
        self.L = L #total hops
        self.d = d #hyperparameter to determine C_{k,out} output channel :type(float)
        self.A = A #adjacency matrix
        self.num_nodes = num_nodes #number of nodes
        self.in_dim = in_dim
        if out_dim == 0: #We do not want the output dimension to change from the input_dim
            self.out_dim = in_dim
        else:
            self.out_dim = out_dim
        self.adj_list = self.__make_adjacency_list(A)
        self.batch_norm = nn.BatchNorm1d(self.out_dim)
        self.Prelu = nn.PReLU()

        #short range
        S_hops_away = []
        for i in range(num_nodes): #for all nodes
            collector = []
            collector_plus = [i] #We need to ensure it considers itself as well
            adj_list = copy.deepcopy(self.adj_list)
            self.__find_k_hops(i,S,collector,collector_plus,adj_list)#check the number of nodes that are at most S-hops away from i
            S_hops_away.append(collector_plus)

        #for each node, we consider nodes at most S-hops away and create weight matrix of same in and out dimension.
        self.weights_S = nn.ParameterList( [nn.Parameter(torch.FloatTensor(len(S_hops_away[i]),in_dim,in_dim)) for i in range(num_nodes)] ) #check for len(S_hops_away[i] == 0
        #initialize
        for ele in self.weights_S:
            init.xavier_uniform_(ele)
        self.S_hops_away = S_hops_away


        #long range
        L_hops_away = []
        for i in range(num_nodes):
            holder = dict()
            for j in range(S+1,L+1): #We check from S+1 till L (added +1 to L because of python)
                collector = []
                collector_plus = []
                adj_list = copy.deepcopy(self.adj_list)
                self.__find_k_hops(i,j,collector,collector_plus,adj_list)#check the number of nodes that are j away from i
                holder[str(j)] = collector
            L_hops_away.append(holder)

        self.L_hops_away = L_hops_away
        self.c_k_out = dict()
        total_L_dim = 0
        for k in range(S+1,L+1):
            doodle = int(d**(k-L)*in_dim) #just liked the name (dummy variable)
            self.c_k_out[str(k)] = doodle #output dimensions for long range nodes
            total_L_dim += doodle
        self.total_L_dim = total_L_dim #sum of all the dimensions for C_{k_out} to be used in determining dimension of H_a
        self.weight_Wa = nn.Parameter(torch.FloatTensor(self.out_dim+total_L_dim,self.out_dim)) #This would be used to multiply H_a to get H
        #initialize
        init.xavier_uniform_(self.weight_Wa)

        self.weights_L = nn.ParameterDict()
        for k in range(S+1,L+1): #for each k-hop
            for i,node in enumerate(L_hops_away): #run through each node
                if len(node[str(k)]) != 0: #I have to set condition to make sure we do not set L larger than average path length
                    #for each node (in each k-hop) create a weight matrix of the number of nodes k-hops away from node, with the same in channel
                    #and out channel is computed using the expression provided.
                    #for each node, we take the elements that are k-hops away an create a tensor for them
                    self.weights_L[str(k)+'+'+str(i)] = nn.Parameter(torch.FloatTensor(len(node[str(k)]),in_dim,self.c_k_out[str(k)]))
                    #initialize
                    init.xavier_uniform_(self.weights_L[str(k)+'+'+str(i)])
                    # self.weights_L[str(k)+'+'+str(i)] helps find the weights of k-hops away neighbours for each node i

    def __make_adjacency_list(self,A):
        """
            This function creates the adjacency list for A.
            A is a num_nodes x num_nodes adjacency matrix.
        """
        num_nodes = len(A)
        adj_list = dict()
        for i in range(num_nodes):
            #Added False so that I can know which vertex has been visited
            adj_list[str(i)] = [[j,False] for j in range(num_nodes) if int(A[i,j]) != 0]
        return adj_list

    def __find_k_hops(self,node,k,collector,collector_plus,adj_list):
        """
            This function returns the elements that are k-hops away from node.
            collector:  holds nodes that are k-hops away :type(list)
            k:          number of hops we are interested in :type(int)
            node:       the vertex we start from :type(int)
            adj_list:   adjacency list of our graph :type(list)
            collector_plus: all the nodes that are at most k-hops away from node :type(list)
        """
        for ele in adj_list: #Goes through each key
            if int(ele) != node: #we know we don't need to check the node we are in
                for each in adj_list[ele]: #Go through each neighbour of ele
                    if each[0] == node: #if ele is a direct neighbur of node
                        each[1] = True #infom it that node has been visited
                        break #every pair of nodes have just 1 connection. So if we find it once, we do not need to check further (saves small time)

        if k != 0:
            for each in adj_list[str(node)]:
                if not each[-1]: #if not visited
                    collector_plus.append(each[0])
                    self.__find_k_hops(each[0],k-1,collector,collector_plus,adj_list) #we have k-1 because we reduce total hops left by 1 each time we move through edge
        else:
            collector.append(node)


    def forward(self,features,A): #Consider changing A:= M_k + O_k according to paper.
        """
            This function implements the Hierarchical Channel Squeezing Fusion layer.
            features:   feature matrix--dim(batch_size,num_nodes,in_channel)
        """
        batch_size = features.size(0)
        #This is a tensor whose entries contains h_S_i across each batch for all node i
        #The dimension 1 is just to allow it accomodate the returned array of dummy_var
        if features.device.type == 'cuda':
            H_S = torch.FloatTensor(batch_size,self.num_nodes,1,self.in_dim).cuda()
        else: #The dimension 1 is just to allow it accomodate the returned array of dummy_var
            H_S = torch.FloatTensor(batch_size,self.num_nodes,1,self.in_dim).to(features.device)

        for i,elements in enumerate(self.S_hops_away):
            dummy_var = 0
            for j,ind_x in enumerate(elements):
                #.unsqueeze(1) is needed to keep the Dimension as [batch_size,1,in_dim]
                dummy_var += A[:,i,ind_x].unsqueeze(1).unsqueeze(1)*torch.bmm(features[:,ind_x].unsqueeze(1),self.weights_S[i][j].unsqueeze(0).repeat(batch_size,1,1))
                # Note: A[:,i,idx].unsqueeze(1)-- A is batch_size x num_nodes x num_nodes .unsqueeze(1) so dim can be batch_size x 1 x 1
            H_S[:,i] = dummy_var
        H_S = H_S.squeeze(-2) #This is to undo the unnecessary dimension 1 added above.

        if features.device.type == 'cuda':
            H_L = torch.FloatTensor(batch_size,self.num_nodes,1,self.total_L_dim).cuda()
        else: #The dimension 1 is just to allow it accomodate the returned array of H_k
            H_L = torch.FloatTensor(batch_size,self.num_nodes,1,self.total_L_dim).to(features.device)

        for i,each in enumerate(self.L_hops_away):
            for k in range(self.S+1,self.L+1):
                dummy_var = 0
                for j,idx in enumerate(each[str(k)]):
                    #.unsqueeze(1) is needed to keep the Dimension as [batch_size,1,in_dim]
                    dummy_var += A[:,i,idx].unsqueeze(1).unsqueeze(1)*torch.bmm(features[:,idx].unsqueeze(1),self.weights_L[str(k)+'+'+str(i)][j].unsqueeze(0).repeat(batch_size,1,1))
                    # Note: A[:,i,idx].unsqueeze(1)-- A is batch_size x num_nodes x num_nodes .unsqueeze(1) so dim can be batch_size x 1 x 1
                if k == (self.S+1):
                    H_k = dummy_var.squeeze(1) #Ensures dimension of dummy_var is [batch_size,C_{k,out}]
                else:
                    H_k = torch.cat((H_k,dummy_var.squeeze(1)),1) #consider using torch.cat(H_k,0)

            H_L[:,i] = H_k.unsqueeze(1) #.unsqueeze(1) makes it [batch_size,1,total_L_dim]
        H_L = H_L.squeeze(-2) #This is to undo the unnecessary dimension 1 added above.

        H_a = torch.cat((H_S,H_L),dim=-1) #dimension (batch_size,num_nodes,out_dim+total_L_dim)
        H = torch.bmm(H_a,self.weight_Wa.unsqueeze(0).repeat(batch_size,1,1)) #should be (batch_size,num_nodes,out_dim)
        H = self.batch_norm(H.transpose(1,2)).transpose(1,2) #reshape so we can easily use batch norm
        #consider putting a (leaky) ReLU on H ie. F.ReLU(H) to give non-linearity
        H = self.Prelu(H) #for nonlinearity

        return H
