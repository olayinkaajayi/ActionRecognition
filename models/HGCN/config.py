import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'model_config': {
        'task': ('nc', 'which tasks to train on, can be any of [lp, nc]'),
        'model': ('HGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HGCN]'),
        'dim': (16, 'embedding dimension'),
        'manifold': ('Hyperboloid', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'), #formerly==1.0
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'num_layers': (2, 'number of hidden layers in encoder'), #Even though this is 2, we still have 1 layer of GNN
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'use_att': (0, 'whether to use hyperbolic attention or not'),
        'local_agg': (0, 'whether to local tangent space aggregation or not'),
        'dropout': (0.1, 'dropout probability'),
        'cuda': (1, 'which cuda device to use (-1 for cpu training)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
