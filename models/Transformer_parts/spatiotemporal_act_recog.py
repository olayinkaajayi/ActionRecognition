import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from parts_gnn_trans import Parts_GNN_Trans
from body_gnn_trans import Body_GNN_Trans
from joints_gnn_trans import Joints_GNN_Trans

class Spatiotemp_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, in_dim, hidden_size, mlp_numbers, num_trans_layers=2):

        super(Spatiotemp_Action_recog, self).__init__()

        # parts
        # self.parts = Parts_GNN_Trans(in_dim, hidden_size, mlp_numbers, num_trans_layers=num_trans_layers) #, num_parts=10)

        # joints
        self.joints = Joints_GNN_Trans(in_dim, hidden_size, mlp_numbers, num_layers=num_trans_layers) #, num_nodes=25)

        # body
        # self.body = Body_GNN_Trans(in_dim, hidden_size, mlp_numbers, num_trans_layers=num_trans_layers) #, num_gc_layers=2)


    def forward(self, features, A, epoch=None):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        # parts
        # out_parts, s_l, anext = self.parts(features, A)
        # out_parts = self.parts(features, A)

        # joints
        out_joints = self.joints(features, A)
        scores = out_joints # added
        # body
        # out_body, g_enc, l_enc, neg_enc = self.body(features, A)

        # scores = out_body + out_joints + out_parts
        # scores = out_parts
        # g_enc, l_enc, neg_enc = torch.tensor([0],device=features.device), torch.tensor([0],device=features.device), torch.tensor([0],device=features.device)
        # s_l = torch.tensor([0],device=features.device)
        return scores#, g_enc, l_enc, neg_enc, s_l
