import torch
import torch.nn as nn

from interact import Interact, FeedForwardNetwork
# from gcn_model_plus_learn import ImpConGCN as GCN
from gcn_model import GCN
# from transformer import Transformer
from transformer_no_pool import Transformer
# from temporal_fourier_res import TemporalEncoder


class MultiStream(nn.Module):
    """
        The MultiStream takes into account the interaction between 2 skeletons,
        and the action of each individual skeleton.
    """

    def __init__(self, hidden_size, num_nodes=25, layer1=False, use_intr=True):
        super(MultiStream, self).__init__()

        # Spatial modules
        if use_intr:
            self.interact = Interact(hidden_size, hidden_size)
        else:
            self.interact = None

        self.dropout_interact = nn.Dropout(0.1)
        self.gnn1 = GCN(hidden_size, hidden_size, use_bn=False, layer1=layer1)
        self.gnn2 = GCN(hidden_size, hidden_size, use_bn=False, layer1=layer1)
        self.mlp_conn1 = FeedForwardNetwork(2*hidden_size, hidden_size)
        self.mlp_conn2 = FeedForwardNetwork(2*hidden_size, hidden_size)

        # Temporal modules
        # self.temporal1 = TemporalEncoder(hidden_size=num_nodes*hidden_size)
        # self.temporal2 = TemporalEncoder(hidden_size=num_nodes*hidden_size)
        self.temporal1 = Transformer(hidden_size=num_nodes*hidden_size,
                                    filter_size=num_nodes*hidden_size,
                                    num_nodes=num_nodes)
        self.temporal2 = Transformer(hidden_size=num_nodes*hidden_size,
                                    filter_size=num_nodes*hidden_size,
                                    num_nodes=num_nodes)


    def forward(self, x1, x2, A=None):

        stream1 = self.gnn1(x1, A) #[b x time x n_nodes x filter_size]
        stream2 = self.gnn2(x2, A) #[b x time x n_nodes x filter_size]

        if self.interact is not None:
            interact = self.interact(x1, x2) #[b x time x n_nodes x filter_size]
            interact = self.dropout_interact(interact)

            stream1 = self.mlp_conn1(torch.concat([stream1,interact], dim=-1)) #[b x time x n_nodes x filter_size]
            stream2 = self.mlp_conn2(torch.concat([stream2,interact], dim=-1)) #[b x time x n_nodes x filter_size]

        stream1 = self.temporal1(stream1) #[b x time x n_nodes x filter_size]
        stream2 = self.temporal2(stream2) #[b x time x n_nodes x filter_size]

        return x1 + stream1, x2 + stream2
