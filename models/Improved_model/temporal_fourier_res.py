import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fourier_res import FourierResistor

class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, head_size, dropout_rate=0.1):
        super(Encoder, self).__init__()
        # filter_size: would MLP hidden size to pass the result of X+inv_fourier
        self.fourier_res = FourierResistor(hidden_size, head_size, dropout_rate)

    def forward(self, x, pos):
        y = self.fourier_res(x, pos) #shape: b x N x d
        return y

############################################################################
##Consider increasing head_size to 8, also see the effect of reducing head_size
class TemporalEncoder(nn.Module):
    def __init__(self,
                 hidden_size=64, # would be num_nodes*out_dim
                 filter_size=128, #Not needed for now
                 num_nodes=25,
                 head_size=4,
                 dropout_rate=0.1):
        super(TemporalEncoder, self).__init__()

        self.num_nodes = num_nodes

        self.hidden_size = hidden_size #size of each embedding vector

        self.i_emb_dropout = nn.Dropout(dropout_rate)

        self.encoder = Encoder(hidden_size, filter_size,
                               head_size, dropout_rate)

        # For positional encoding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, inputs):
        # inputs: batch_size x time x num_nodes x out_channel
        batch_size, time_dim, _, _ = inputs.shape
        inputs = inputs.reshape(batch_size,time_dim,-1) # [b x time x num_nodes*out_channel]
        enc_output = self.encode(inputs) # [b, q_len, filter_size] q_len == time
        enc_output = enc_output.reshape(batch_size,time_dim,self.num_nodes,-1) # [b x time x num_nodes x out_channel]
        return enc_output

    def encode(self, inputs):

        input_embedded = inputs
        pos = self.get_position_encoding(inputs)

        return self.encoder(inputs, pos)


    def get_position_encoding(self, x):
        batch_size = x.size(0)
        max_length = x.size(1)
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)
        signal = F.pad(signal, (0, 0, 0, self.hidden_size % 2))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal.repeat(batch_size,1,1) #Note: signal.shape: torch.Size([batch_size, 300, 64])
