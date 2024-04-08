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

    def __init__(self, in_dim, hidden_size, mlp_numbers, gnn_model, num_trans_layers=2, starting_epoch=40, rel_err=0.25):

        super(Spatiotemp_Action_recog, self).__init__()
        self.starting_epoch = starting_epoch
        self.rel_err = rel_err
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

        self.joints = Joints_GNN_Trans(hidden_size, hidden_size, mlp_numbers,
                                        num_layers=num_trans_layers, new_strategy=True, democracy=True) #, num_nodes=25)


    def forward(self, features, A, epoch=0):
        """
            features:   --dim(batch_size,time,num_nodes,in_channel)
            A:          --dim(num_nodes,num_nodes)
        """

        # body
        out_body = []
        out = features
        for i in range(self.num_gc_layers):
            out = F.relu( self.gnn_body[i]( out, A ) )
            out_body.append( out )

        # joints
        joints_input = torch.cat( ( features , torch.cat(out_body,dim=-1) ), dim=-1)
        joints_input = self.compress(joints_input)
        scores, candidates = self.joints(joints_input, A, reached_starting_epoch= ((epoch+1) >= self.starting_epoch))
        # candidates: [b x time x n_nodes x filter_size]
        # scores: [b x num_class]
        return scores, *self.select_democrats(scores, candidates)


    def select_democrats(self, scores, candidates):
        """This function selects the positive and negative samples for the highly correlated classes"""
        if candidates is None:
            return [None]*6 #since we are going to have 6 outputs


        topk_vals , topk_idx = scores.topk(2,dim=1,largest=True,sorted=True)

        rel_err = (topk_vals[:,0] - topk_vals[:,1]).abs() / (topk_vals.max(dim=1)[0]).abs()

        class_of_closest_pair = topk_idx[rel_err <= self.rel_err] #[#closest_pair x 2]

        class_of_further_pair = topk_idx[rel_err > self.rel_err] #[#further_pair x 2]

        samples_of_closest_pairs = (rel_err <= self.rel_err).nonzero().squeeze(-1) #[#closest_pairs]

        samples_of_further_pairs = (rel_err > self.rel_err).nonzero().squeeze(-1) #[#further_pairs]

        return [candidates[samples_of_further_pairs], class_of_further_pair, samples_of_further_pairs,
                candidates[samples_of_closest_pairs], class_of_closest_pair, samples_of_closest_pairs]
