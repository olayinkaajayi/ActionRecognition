import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from parts_gnn_trans import Parts_GNN_Trans
from gcn_model import GCN
from joints_gnn_trans import Joints_GNN_Trans

class Spatiotemp_Action_recog(nn.Module):
    """This module implements a skeletal action recognition model using the vanila GCN model
        and a fully connected layer.
    """

    def __init__(self, in_dim, hidden_size, mlp_numbers, gnn_model, weighted_combine=False, num_trans_layers=2):

        super(Spatiotemp_Action_recog, self).__init__()

        # parts
        self.parts = Parts_GNN_Trans(in_dim, hidden_size, mlp_numbers, num_trans_layers=num_trans_layers) #, num_parts=10)

        # body
        self.num_gc_layers = gnn_model.num_gc_layers
        self.gnn_body = nn.ModuleList()

        with torch.no_grad():
            for i in range(gnn_model.num_gc_layers):
                feature_in_dim = (gnn_model.dataset_num_features) if i==0 else (gnn_model.hidden_dim)
                self.gnn_body.append(GCN(feature_in_dim , gnn_model.hidden_dim))
                self.gnn_body[i].conv.weight.copy_(gnn_model.encoder.convs[i].lin.weight)
                self.gnn_body[i].conv.weight.requires_grad = False

        # joints
        joints_in_dim = (gnn_model.hidden_dim * gnn_model.num_gc_layers) + gnn_model.dataset_num_features
        ########################################################################################################
        self.compress = nn.Linear(joints_in_dim,hidden_size) #consider increasing the output dimension of compress

        self.joints = Joints_GNN_Trans(hidden_size, hidden_size, mlp_numbers, num_layers=num_trans_layers, new_strategy=True) #, num_nodes=25)

        #combine with weights???
        _,output_dim = mlp_numbers
        self.weighted_combine = weighted_combine
        if weighted_combine:
            self.combine_scores = nn.Sequential(nn.Linear(2*output_dim,output_dim),
                                                nn.ReLU(),
                                                nn.Linear(output_dim,output_dim))

    def forward(self, features, A):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """
        # parts
        # out_parts, s_l, anext = self.parts(features, A)
        # out_parts = self.parts(features, A)

        # body
        out_body = []
        out = features
        for i in range(self.num_gc_layers):
            out = F.relu( self.gnn_body[i]( out, A ) )
            out_body.append( out )

        # joints
        joints_input = torch.cat( ( features , torch.cat(out_body,dim=-1) ), dim=-1)
        joints_input = self.compress(joints_input)
        out_joints = self.joints(joints_input, A)


        if self.weighted_combine:
            scores = self.combine_scores(torch.cat((out_joints , out_parts), dim=-1))
        else:
            scores = out_joints #+ out_parts

        return scores
