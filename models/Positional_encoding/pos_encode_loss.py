import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Position_encode_loss(torch.nn.Module):
    """The loss function for the positional encoding."""

    def __init__(self, k1=1.0, k2=1.0):
        super(Position_encode_loss, self).__init__()

        self.k1 = k1 #hyperparameter
        self.k2 = k2 #hyperparameter

    def forward(self, *args):
        """Loss for the proposed algorithm"""
        L_deg, L_adj, L_deg_dist = args

        return L_adj + self.k1*L_deg_dist + self.k2*L_deg
