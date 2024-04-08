import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=4, att_size=None, num_frames=300):
        super(MultiHeadAttention, self).__init__()

        att_size = hidden_size if att_size is None else att_size
        self.att_size = att_size
        self.head_size = head_size

        self.num_frames = num_frames

        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size*att_size, bias=False)

        self.linear_k = nn.Linear(hidden_size, head_size*att_size, bias=False)

        self.linear_v = nn.Linear(hidden_size, head_size*att_size, bias=False)

        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        # We need this output layer, as it serves as our filter by allowing
        # only important information pass through it.
        self.output_layer = nn.Linear(att_size * head_size, hidden_size,
                                      bias=False) #we might need to add a ReLU in this layer
        initialize_weight(self.output_layer)


    def forward(self, q, k, v,cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # may be forced to loop to compute q, k & v of each attention head
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)

        q = self.linear_q(q).view(batch_size, -1, self.num_frames, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_frames, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_frames, self.head_size, d_k)

        q = q.transpose(-2, -3)   # [b, l, h, n_frame, d_k]
        v = v.transpose(-2, -3)   # [b, l, h, n_frame, d_k]
        k = k.transpose(-2, -3).transpose(-1, -2)  # [b, l, h, d_k, n_frame]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, l, h, n_frame, n_frame]
        x = torch.softmax(x, dim=-1)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, l, h, n_frame, attn]

        x = x.transpose(2, 3).contiguous()  # [b, l, n_frame, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v) # [b, q_len, h*d_v]

        x = self.output_layer(x) # [b, q_len, hidden_size]

        assert x.size() == orig_q_size
        return x



class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, head_size, att_size=None, num_frames=1):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate, head_size, att_size, num_frames)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        # if num_frames==300:
        #     kernel_size = 3
        #     stride = 3
        #
        # elif num_frames==150:
        #     kernel_size = 3
        #     stride = 2
        #
        # else:
        #     kernel_size = num_frames
        #     stride = num_frames

        # self.gather_imp = Conv_n_Bnorm(hidden_size,hidden_size,
        #                                 kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y) # [b, q_len, hidden_size]
        y = self.self_attention_dropout(y)
        # Add self.gather_imp here before the skip
        # y = self.gather_imp(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y) # [b, q_len, filter_size]
        y = self.ffn_dropout(y)
        x = x + y
        x = self.last_norm(x)
        return x# self.gather_imp(x)


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, head_size, num_frames,
                    att_size=None, max_time_len=300, scale_factor=4):
        super(Encoder, self).__init__()

        red_size = hidden_size//scale_factor
        self.reduce_dim = Conv_n_Bnorm(hidden_size,red_size)
        self.increase_dim = Conv_n_Bnorm(red_size,hidden_size)

        num_frames = num_frames if len(num_frames) != 0 else [max_time_len]
        encoders = [EncoderLayer(red_size, red_size, dropout_rate, head_size, att_size, n_frame)
                    for n_frame in num_frames]
        self.layers = nn.ModuleList(encoders)


    def forward(self, inputs):
        inputs = self.reduce_dim(inputs)
        total_encoder_output = 0
        for i,enc_layer in enumerate(self.layers): #May need more than 1 layer, as this serves as a macro-view of the temporal domain
            encoder_output = enc_layer(inputs)
            total_encoder_output += encoder_output

        return self.increase_dim(total_encoder_output)


class Conv_n_Bnorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, seq_len=300):
        super(Conv_n_Bnorm, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride)

        self.bn = nn.BatchNorm1d(out_channels)
        self.L_out = self.get_output_length_1d_cnn(seq_len, kernel_size, stride)


    def get_output_length_1d_cnn(self, input_length, kernel_size, stride=1, padding=0, dilation=1):
        """
        Calculates the length of the output sequence for a 1D CNN layer in PyTorch.
        """
        output_length = ((input_length + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
        return int(output_length)


    def forward(self, x):
        # x: [b, time_dim, num_nodes*hidden_size]
        x = self.conv(x.transpose(1,2)) #[b, conv_size, time_dim]
        x = self.bn(x)
        return x.transpose(1,2) #[b, time_dim, conv_size]



class Transformer(nn.Module):
    def __init__(self,
                 hidden_size=64, # would be num_nodes*out_dim
                 filter_size=64, # This helps us dropout parts
                 num_nodes=25,
                 dropout_rate=0.1,
                 att_size=None,
                 head_size=4,
                 num_frames=[6, 30, 150, 300],
                 max_time_len=300):
        super(Transformer, self).__init__()

        self.num_nodes = num_nodes

        self.hidden_size = hidden_size #size of each embedding vector

        self.i_emb_dropout = nn.Dropout(dropout_rate)

        self.encoder = Encoder(hidden_size, filter_size,
                               dropout_rate, head_size,
                               num_frames,att_size=att_size,
                               max_time_len=max_time_len)

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

    def encode(self, inputs): #remove i_mask
        # Input embedding
        # Input would be the GNN output of shape #batch_size x time_dim x out_dim_GCN
        input_embedded = inputs + self.get_position_encoding(inputs)
        input_embedded = self.i_emb_dropout(input_embedded)

        return self.encoder(input_embedded)


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
