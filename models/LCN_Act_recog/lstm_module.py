import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class Lstm_cell(nn.Module):
    """
        This model runs the LSTM function for our model.
    """

    def __init__(self, input_dim,output_dim,hidden_dim=0):
        super(Lstm_cell, self).__init__()
        self.input_dim = input_dim
        if hidden_dim == 0:
            self.hidden_dim = int(input_dim * 1.5) #Just to make the hidden_dim bigger by 50%
        else:
            self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm_cell = nn.LSTMCell(input_dim, self.hidden_dim)
        init.xavier_uniform_(self.lstm_cell.weight_hh)
        init.xavier_uniform_(self.lstm_cell.weight_ih)
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, output_dim) #This should map to the number of classes in the data
        init.xavier_uniform_(self.fc.weight)

    def forward(self, input_sample):
        #Verify dimension of input_sample
        """
            input_sample: --dim(time,batch_size,input_dim)
        """
        batch_size = input_sample.size(1)
        time = input_sample.size(0)
        if input_sample.device.type == 'cuda':
            h_t = torch.zeros(batch_size, self.hidden_dim).cuda()
            c_t = torch.zeros(batch_size, self.hidden_dim).cuda()
        else:
            h_t = torch.zeros(batch_size, self.hidden_dim).to(input_sample.device)
            c_t = torch.zeros(batch_size, self.hidden_dim).to(input_sample.device)

        for input_t in input_sample.split(time, dim= 0)[0]: #dim should be the time dimension
            # input_t : batch_size X input_size
            h_t, c_t = self.lstm_cell(input_t, (h_t, c_t))

        return self.fc(h_t)
