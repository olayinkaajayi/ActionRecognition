import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Lstm_block(nn.Module):
    """
        This model runs the LSTM function for our model.
    """

    def __init__(self, input_dim,hidden_dim=0):
        super(Lstm_block, self).__init__()
        self.input_dim = input_dim
        if hidden_dim == 0:
            self.hidden_dim = int(input_dim * 1.5) #Just to make the hidden_dim bigger by 50%
        else:
            self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim,batch_first=True)
        # init.xavier_uniform_(self.lstm.all_weights)

    def forward(self, input_sample):
        """
            input_sample: --dim(batch_size,time,input_dim)
        """
        batch_size = input_sample.size(0)
        if input_sample.device.type == 'cuda':
            h_t = torch.zeros(1,batch_size, self.hidden_dim).cuda()
            c_t = torch.zeros(1,batch_size, self.hidden_dim).cuda()
        else:
            h_t = torch.zeros(1,batch_size, self.hidden_dim).to(input_sample.device)
            c_t = torch.zeros(1,batch_size, self.hidden_dim).to(input_sample.device)

        # input_t : batch_size X time X input_size
        output , (h_t, c_t) = self.lstm(input_sample, (h_t, c_t))

        return h_t.squeeze(0) #shape: batch_size x hidden_dim
