from torch.nn import Sequential, Linear, ReLU
from encoder import Body_GNN as Encoder
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np


class InfoGraph(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim, num_gc_layers,dropout_rate=0.2):
        super(InfoGraph, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_gc_layers = num_gc_layers
        self.dataset_num_features = dataset_num_features
        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.dropout_rate = dropout_rate #useful for negative embeddings

        self.proj_global = Linear(self.embedding_dim, self.embedding_dim)

        self.init_emb()


    def init_emb(self):
        """
            Initialize the weights of the embeddings.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, data, A= None):
        if A is None:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        else:
            x, edge_index = data, A

        g_enc, l_enc = self.encoder(x, edge_index)

        # For negative samples
        batch_size = torch.max(data.batch).item() + 1 if A is None else x.shape[0]
        neg_sample = x.reshape(batch_size,-1,self.dataset_num_features) if A is None else x[:,:,:,:] # -1 is number of nodes
        nb_nodes = neg_sample.shape[-2] #should be 25
        idx = np.random.permutation(nb_nodes)
        # idx = list(range(nb_nodes))
        neg_sample = neg_sample[:, idx, :].reshape(-1,self.dataset_num_features) if A is None else neg_sample[:, :, idx, :]  #might shuffle dataset_num_features too
        neg_sample = nn.functional.dropout(neg_sample,p=self.dropout_rate)

        neg_enc = self.encoder(neg_sample, edge_index, get_global=False)

        return self.proj_global(g_enc), l_enc, neg_enc

def loss_fn(pos_z, neg_z, summary):
    r"""Computes the mutual information maximization objective."""
    EPS = 1e-15 # tolerance

    pos_loss = -torch.log(
        discriminate(pos_z, summary, sigmoid=True) + EPS).mean(dim=1) # b
    neg_loss = -torch.log(1 -
                          discriminate(neg_z, summary, sigmoid=True) +
                          EPS).mean(dim=1) # b
    #The negative sign is because we wish to maximize the mutual information
    return -torch.sum(pos_loss + neg_loss) #added a minus

def discriminate(z, summary, sigmoid=True):
    r"""Given the patch-summary pair :obj:`z` and :obj:`summary`, computes
    the probability scores assigned to this patch-summary pair.

    Args:
        z (Tensor): The latent space.
        summary (Tensor): The summary vector.
        sigmoid (bool, optional): If set to :obj:`False`, does not apply
            the logistic sigmoid function to the output.
            (default: :obj:`True`)
    """
    # summary : b x dim
    # z : b x n_nodes x dim
    n_nodes = 25
    num_features = z.shape[-1]

    matx2 = summary.unsqueeze(1).transpose(1,2) # b x dim x 1
    z = z.reshape(-1,n_nodes,num_features) # b x n_nodes x dim
    value = torch.bmm(z, matx2).squeeze(-1) # b x n_nodes
    return torch.sigmoid(value) if sigmoid else value # b x n_nodes


def validation_fn(model, loader, device, many_gpu, batch_size):
    """
        This function returns the loss of the validation/test data.
    """
    pbar = tqdm(loader, unit='batch')
    loss_all = 0
    average_loss = 0.
    model.eval()
    with torch.no_grad():
        for data in pbar:
            data = data if many_gpu else data.to(device)
            g_enc, l_enc, neg_enc = model(data)

            # measure='JSD'
            # loss = local_global_loss_(l_enc, g_enc, neg_enc, measure)
            loss = loss_fn(l_enc,neg_enc,g_enc)
            loss_all += loss.item() #* num_graphs
            # report
            pbar.set_description('Val')
        average_loss = loss_all / len(loader)

    return average_loss
