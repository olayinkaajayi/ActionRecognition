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


class MultiHeadAttention(nn.Module): # we set head_size=25 so that each node is handled individually
    def __init__(self, hidden_size, dropout_rate, head_size=25, divide=True):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        value = (hidden_size // head_size) if divide else hidden_size
        self.att_size = att_size = value
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)



    def forward(self, q, k, v,cache=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # may be forced to loop to compute q, k & v of each attention head
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)

        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v) # [b, q_len, h*d_v]

        x = self.output_layer(x) # [b, q_len, hidden_size]

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, head_size, divide):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate,head_size,divide)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y) # [b, q_len, hidden_size]
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y) # [b, q_len, filter_size]
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, head_size, divide):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, filter_size, dropout_rate, head_size, divide)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs):
        encoder_output = inputs
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output)
        return self.last_norm(encoder_output)



class Transformer(nn.Module):
    def __init__(self,
                 n_layers=1, #n_layers should be 1 (change to 2 or more)
                 hidden_size=64, # would be num_nodes*out_dim
                 filter_size=64, # same size as hidden_size
                 num_nodes=25,
                 divide=True,
                 pool=False,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.num_nodes = num_nodes

        self.hidden_size = hidden_size #size of each embedding vector

        self.i_emb_dropout = nn.Dropout(dropout_rate)

        self.encoder = Encoder(hidden_size, filter_size,
                               dropout_rate, n_layers,head_size=num_nodes,divide=divide) #n_layers should be 1

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
