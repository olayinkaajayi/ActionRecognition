import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune

from training_utils_py2 import use_cuda

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
    def __init__(self, hidden_size, dropout_rate, head_size=1, mask=None, att_size=None):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size
        self.extra_head = 4 # hand of hands

        if att_size is None: #the att_size contributes as a multi-head attention for the "node, body or part" of interest
            att_size = (hidden_size // head_size) // self.extra_head #added
        self.att_size = att_size // self.extra_head #added
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * self.extra_head * self.att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_size * self.extra_head * self.att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_size * self.extra_head * self.att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.mask = mask #This is to ensure a non-fully connected FFN

        self.att_dropout = nn.Dropout(dropout_rate)
        # output_layer would be needing some masking
        self.output_layer = nn.Linear(head_size * self.extra_head * self.att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)


    def forward(self, q, k, v,cache=None):
        # mask the relevant weights
        print("Mask is None" if self.mask is None else "Mask is not None")
        print("shape:", isinstance(self.linear_q.weight,nn.Parameter))
        print("shape:", isinstance(self.linear_k.weight,nn.Parameter))
        print("shape:", isinstance(self.linear_v.weight,nn.Parameter))
        print("\n\n")
        if self.mask is not None:
            prune.custom_from_mask(self.linear_q, name='weight', mask=self.mask)
            prune.custom_from_mask(self.linear_k, name='weight', mask=self.mask)
            prune.custom_from_mask(self.linear_v, name='weight', mask=self.mask)
        # masking done
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        # you may consider adding another channel as head for each of the parts or nodes you'll be considering
        # so we would have something like q = .view(batch_size, -1, self.head_size, extra_head, d_k//extra_head)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, self.extra_head, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, self.extra_head, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, self.extra_head, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2).transpose(2, 3)                   # [b, h, ex_h, q_len, d_k]
        v = v.transpose(1, 2).transpose(2, 3)                   # [b, h, ex_h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3).transpose(3, 4)   # [b, h, ex_h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        # if we use extra_head here, then the attention would be for each head seperately
        # and we will have something like [b, h, ex_head, q_len, k_len] (hands of hands)
        x = torch.matmul(q, k)  # [b, h, ex_h, q_len, k_len]
        x = torch.softmax(x, dim=-1) # [b, h, ex_h, q_len, k_len]
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, ex_h, q_len, attn]

        x = x.transpose(2, 3).transpose(1, 2).contiguous()  # [b, q_len, h, ex_h, attn]
        x = x.view(batch_size, -1, self.head_size * self.extra_head * d_v) # [b, q_len, h*ex_h*d_v]

        if self.mask is not None: # output_layer needs a transposed mask since the reverse is happening
            prune.custom_from_mask(self.output_layer, name='weight', mask=self.mask.t())
        x = self.output_layer(x) # [b, q_len, hidden_size]

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate,head_size=1,mask=None,att_size=None):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate,head_size,mask,att_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x,holder=None,hold=False):
        y = self.self_attention_norm(x)
        if hold and (holder is not None): #appends for i (from caller) = 1 & 2
            #We need this because we wish to use only normalized values
            holder.append(y)#holds the output of each encoder (body, joints and parts)
        y = self.self_attention(y, y, y) # [b, q_len, hidden_size]
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y) # [b, q_len, filter_size]
        y = self.ffn_dropout(y)
        x = x + y
        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate, n_layers, att_size=None, use_sum=True, order_type=1):
        super(Encoder, self).__init__() # head_size=25 is num_nodes
        ##########ADDED##############
        extra_head = 4 #hand of hands
        att_size = (filter_size // 25) if att_size is None else att_size
        att_size = att_size * extra_head
        # body_mask = None # Do not really need this
        joints_mask = get_joint_mask(filter_size,head_size=25,att_size_out=att_size)
        # parts_att_size = (filter_size // 25) if att_size is None else att_size #25= num_nodes
        # parts_mask = get_part_mask(filter_size,head_size=12,att_size_out=parts_att_size) # head_size is 12 as there are 12 parts

        body_encoder = EncoderLayer(hidden_size, filter_size, dropout_rate,att_size=att_size)
        # parts_encoder = EncoderLayer(hidden_size, filter_size, dropout_rate,head_size=12,att_size=parts_att_size,mask=parts_mask)
        joints_encoder = EncoderLayer(hidden_size, filter_size, dropout_rate,head_size=25,att_size=att_size,mask=joints_mask)

        if order_type == 1:
            encoders = [body_encoder,joints_encoder]
        else:
            encoders = [joints_encoder,body_encoder]

        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.use_sum = use_sum
        if not use_sum:
            combine_weight = [nn.Linear(filter_size,filter_size,bias=False) for i in range(len(encoders))]
            self.combine_weight = nn.ModuleList(combine_weight)

            for ele in self.combine_weight:
                initialize_weight(ele)

    def forward(self, inputs):
        encoder_output = inputs
        holder = [] #holds the output of each encoder (body, joints and parts)

        for i,enc_layer in enumerate(self.layers):
            hold = False if i==0 else True
            encoder_output = enc_layer(encoder_output,holder,hold)
        encoder_output = self.last_norm(encoder_output)
        holder.append(encoder_output)

        encoder_output = torch.zeros_like(encoder_output)
        for i,entry in enumerate(holder):
            encoder_output += entry if self.use_sum else self.combine_weight[i](entry)

        return encoder_output # [b, q_len,filter_size]


class Transformer(nn.Module):
    def __init__(self,
                 n_layers=1, #n_layers should be 1 (change to 2 or more)
                 hidden_size=64, # would be num_nodes*out_dim
                 filter_size=64, # same size as hidden_size
                 num_nodes=25,
                 pool=False,
                 use_sum=True,
                 order_type=1, #order matters. order 1 {45%} is better (in 10 epochs). order 2 gave {41%}
                 att_size=None,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.num_nodes = num_nodes

        self.hidden_size = hidden_size #size of each embedding vector

        self.i_emb_dropout = nn.Dropout(dropout_rate)

        self.encoder = Encoder(hidden_size, filter_size,
                               dropout_rate, n_layers, att_size, use_sum, order_type) #n_layers should be 1

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


def get_joint_mask(filter_size,head_size,att_size_out=None):
    """This function is used to enforce 'joints' in the transformer architecture
        for the human skeletal graph.
    """
    att_size_in = filter_size // head_size
    if att_size_out is None:
        att_size_out = att_size_in

    mask = torch.zeros(filter_size, head_size*att_size_out,device=decide_gpu())
    for i in range(filter_size):
        mask[i*att_size_in:(i+1)*att_size_in, i*att_size_out:(i+1)*att_size_out] = 1
    return mask.t() #make sure you send back transpose of mask


def get_part_mask(filter_size,head_size=12,att_size_out=None):
    """This function is used to enforce 'parts' in the transformer architecture
        for the human skeletal graph.

        head_size: number of parts: 12
        We will be using:
        right_hand(24,25,12,11), right_arm(11,10,9), scarpula(9,21,5), head_neck(4,3),
        left_hand(22,23,8,7), left_arm(7,6,5), torso(21,2,1), waist(17,1,13),
        right_leg(17,18,19), right_feet(19,20), left_leg(13,14,15) and left_feet(15,16).
    """
    num_nodes = 25
    att_size_in = filter_size // num_nodes
    if att_size_out is None:
        att_size_out = att_size_in
    mask = torch.zeros(filter_size, head_size*att_size_out,device=decide_gpu())
    right_hand=[24,25,12,11]; right_arm=[11,10,9]; scarpula=[9,21,5]; head_neck=[4,3];
    left_hand=[22,23,8,7]; left_arm=[7,6,5]; torso=[21,2,1]; waist=[17,1,13];
    right_leg=[17,18,19]; right_feet=[19,20]; left_leg=[13,14,15]; left_feet=[15,16]

    head_num = 0 # a counter for the head (part) we are currently in
    set_mask(right_hand,mask,att_size_in,att_size_out,head_num) ; set_mask(right_arm,mask,att_size_in,att_size_out,head_num) ; set_mask(scarpula,mask,att_size_in,att_size_out,head_num)
    set_mask(head_neck,mask,att_size_in,att_size_out,head_num) ; set_mask(left_hand,mask,att_size_in,att_size_out,head_num) ; set_mask(left_arm,mask,att_size_in,att_size_out,head_num)
    set_mask(torso,mask,att_size_in,att_size_out,head_num) ; set_mask(waist,mask,att_size_in,att_size_out,head_num) ; set_mask(right_leg,mask,att_size_in,att_size_out,head_num)
    set_mask(right_feet,mask,att_size_in,att_size_out,head_num) ; set_mask(left_leg,mask,att_size_in,att_size_out,head_num) ; set_mask(left_feet,mask,att_size_in,att_size_out,head_num)

    return mask.t() #make sure you send back transpose of mask


def set_mask(arr,mask,att_size_in,att_size_out,head_num):

    for ele in arr:
        ele -= 1
        a , b = ele*att_size_in, ele*att_size_in + att_size_in
        u , v = head_num*att_size_out , head_num*att_size_out + att_size_out
        mask[a:b , u:v]

    head_num += 1

def decide_gpu():
    """A quick fix for our GPU problem."""
    many_gpu = torch.cuda.device_count() > 1 #Decide if we use multiple GPUs or not
    device,_ = use_cuda(use_cpu=False,many=many_gpu,verbose=False)
    return device
