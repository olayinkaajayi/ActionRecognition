"""Base model class."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.cuda() #to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.])).cuda() if args.cuda == 1 else nn.Parameter(torch.Tensor([1.])) #may need to pass this to cuda
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1 #why?? Relates to *** below
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            # According to the paper, we concatenate with zero to map from
            #Euclidean to hyperbolic space.
            o = torch.zeros_like(x) # Here is *** (for above)
            x = torch.cat([o[..., 0:1], x], dim=-1) #formerly dim=1
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)

    def decode(self, h, adj, idx=None): #idx is not needed
        output = self.decoder.decode(h, adj)
        # output is the projection of the hyperbolic embedding to the Euclidean space
        return output

    def forward(self, h, adj):
        b = h.size(0)
        assert b > 1
        out = self.encode(h, adj)
        out = self.decode(out, adj)
        return out
