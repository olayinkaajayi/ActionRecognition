import os
import math
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)



class FeedForwardNetwork(nn.Module): #consider using MLP module for this. It has batchnorm in it.
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        # self.bn = nn.BatchNorm2d(filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, filter_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        # x:[b, t, N, hidden_size]
        x = self.layer1(x)
        # x = self.bn(x.transpose(1,3)).transpose(1,3)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class Interact(nn.Module):
    """
        Uses a cross-attention framework to capture the interaction between
        two entities in a video frame.
    """

    def __init__(self, in_dim, hidden_size, dropout_rate=0.1):
        super(Interact, self).__init__()

        # self.self_attention_norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        # self.self_attention_norm2 = nn.LayerNorm(hidden_size, eps=1e-6)

        self.key1 = nn.Linear(in_dim, hidden_size, bias=False)
        self.query1 = nn.Linear(in_dim, hidden_size, bias=False)
        self.value1 = nn.Linear(in_dim, hidden_size, bias=False)

        self.key2 = nn.Linear(in_dim, hidden_size, bias=False)
        self.query2 = nn.Linear(in_dim, hidden_size, bias=False)
        self.value2 = nn.Linear(in_dim, hidden_size, bias=False)

        initialize_weight(self.key1)
        initialize_weight(self.query1)
        initialize_weight(self.value1)
        initialize_weight(self.key2)
        initialize_weight(self.query2)
        initialize_weight(self.value2)

        self.att_dropout1 = nn.Dropout(dropout_rate)
        self.att_dropout2 = nn.Dropout(dropout_rate)

        self.output_layer = FeedForwardNetwork(2*hidden_size, hidden_size)



    def forward(self, x1, x2,cache=None):
        # x1 = self.self_attention_norm1(x1)
        # x2 = self.self_attention_norm2(x2)

        q1 = self.query1(x1) # [b, t, N, hidden_size]
        k1 = self.key1(x1) # [b, t, N, hidden_size]
        v1 = self.value1(x1) # [b, t, N, hidden_size]

        q2 = self.query2(x2) # [b, t, N, hidden_size]
        k2 = self.key2(x2) # [b, t, N, hidden_size]
        v2 = self.value2(x2) # [b, t, N, hidden_size]

        k2 = k2.transpose(2, 3)  # [b, t, hidden_size, N]
        k1 = k1.transpose(2, 3)  # [b, t, hidden_size, N]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        scale = x1.shape[-1] ** -0.5
        q1.mul_(scale)
        q2.mul_(scale)

        x1 = torch.matmul(q1, k2)  # [b, t, N, N]
        x1 = torch.softmax(x1, dim=3)
        x1 = self.att_dropout1(x1)
        x1 = x1.matmul(v1)  # [b, t, N, hidden_size]

        x2 = torch.matmul(q2, k1)  # [b, t, N, N]
        x2 = torch.softmax(x2, dim=3)
        x2 = self.att_dropout2(x2)
        x2 = x2.matmul(v2)  # [b, t, N, hidden_size]

        # You can sum up x1 and x2 as an effect of their interaction
        x = torch.concat([x1,x2],dim=-1) # [b, t, N, 2*hidden_size]

        x = self.output_layer(x) # [b, t, N, hidden_size]

        return x
