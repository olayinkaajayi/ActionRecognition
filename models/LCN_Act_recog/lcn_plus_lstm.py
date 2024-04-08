import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import sys
sys.path.append("DHCS_implement/models")
from LCN_Act_recog.lstm_module import Lstm_cell
from LCN_Act_recog.lcn import Local_con_net

class LCN_p_LSTM(nn.Module):
    """
        This class would combine the LCN and LSTM modules to form our model for the video data.
    """

    def __init__(self, num_nodes, in_dim, num_class,device,out_dim=0,hidden_dim=0):
        super(LCN_p_LSTM, self).__init__()
        self.in_dim = in_dim
        self.num_nodes = num_nodes
        self.num_class = num_class
        self.out_dim = out_dim
        self.device = device
        self.lcn1 = Local_con_net(num_nodes=self.num_nodes,in_dim=self.in_dim,out_dim=self.out_dim,batch_Norm=True)
        self.lcn2 = Local_con_net(num_nodes=self.num_nodes,in_dim=self.in_dim,out_dim=self.out_dim)
        self.hidden_dim = hidden_dim
        self.lstm = Lstm_cell(input_dim=num_nodes,output_dim=num_class,hidden_dim=self.hidden_dim)

    def forward(self,data,A):
        """
            data:   --dim(batch_size,time,num_nodes,in_channel)
            A:      --dim(num_nodes,num_nodes)
        """
        timeLength = data.size(1)
        #create array to hold each time step of LCN
        holder = []
        #run a for loop for each frame of the video with LCN
        #append the result of each iteration to the array
        for i in range(timeLength):
            out = self.lcn1(data[:,i],A,pool=False)
            out = self.lcn2(out,A).unsqueeze(0) #Makes dimension 1 x batch_size x num_nodes
            holder.append(out) #consider changing this using list().append() and then torch.cat(holder,0) when done.
        if data.device.type == 'cuda':
            holder = torch.cat(holder,0).cuda()
        else:
            holder = torch.cat(holder,0).to(data.device)
        #holder dimension is now time x batch_size x num_nodes

        #take the result of the last step and pass to the lstm module
        out = self.lstm(holder.squeeze(-1)) #.squeeze(-1) to squeeze out the unnecessary last dimension.
        #return final result to the calling model.
        return out
