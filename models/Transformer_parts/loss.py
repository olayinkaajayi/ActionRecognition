import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, s_l):

        _, _, nodes, nnext = s_l.shape
        s_l = s_l.reshape(-1, nodes, nnext) #We do this because of the time domain

        entropy = (
            (torch.distributions.Categorical(probs=s_l).entropy())
            .sum(-1)
            .mean(-1)
        )
        assert not torch.isnan(entropy)
        return entropy


class LinkPredLoss(nn.Module):
    def forward(self, adj, s_l):

        _, _, nodes, nnext = s_l.shape
        s_l = s_l.reshape(-1, nodes, nnext) #We do this because of the time domain
        if len(adj.shape) > 2:
            adj = adj.reshape(-1, nnext, nnext) #We do this because of the time domain

        link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(
            dim=(1, 2)
        )
        link_pred_loss = link_pred_loss / (adj.size(-1) * adj.size(-2))
        return link_pred_loss.mean()


class JansonShannon(nn.Module):
    def forward(self, *args):
        g_enc, l_enc, neg_enc = args
        batch_s, time_dim, n_nodes, embed_dim = l_enc.shape

        g_enc = g_enc.reshape(-1,embed_dim) #We do this because of the time domain
        l_enc = l_enc.reshape(-1,n_nodes,embed_dim) #We do this because of the time domain
        neg_enc = neg_enc.reshape(-1,n_nodes,embed_dim) #We do this because of the time domain

        return self.loss_fn(l_enc,neg_enc,g_enc)


    def loss_fn(self, pos_z, neg_z, summary):
        r"""Computes the mutual information maximization objective."""
        EPS = 1e-15 # tolerance

        pos_loss = -torch.log(
            self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean(dim=1) # b
        neg_loss = -torch.log(1 -
                              self.discriminate(neg_z, summary, sigmoid=True) +
                              EPS).mean(dim=1) # b
        #The negative sign is because we wish to maximize the mutual information
        return -torch.sum(pos_loss + neg_loss) #added a minus


    def discriminate(self, z, summary, sigmoid=True):
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


class Combined_loss(torch.nn.Module):
    def __init__(self, k1=1, k2=1, k3=1):
        super(Combined_loss, self).__init__()
        self.k1 = k1 #constant to control the contribution of the JansonShannon loss to the overall model
        self.k2 = k2 #constant to control the contribution of the Linkprediction loss to the overall model
        self.k3 = k3 #constant to control the contribution of the Entropy loss to the overall model
        self.JS = JansonShannon()
        # self.entropy = EntropyLoss()
        # self.link_pred = LinkPredLoss()
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, *args):

        out, label, g_enc, l_enc, neg_enc, s_l, adj = args
        # out, label, g_enc, l_enc, neg_enc = args

        return (
                self.cross_entropy(out,label)
                + self.k1 * self.JS(g_enc, l_enc, neg_enc)
                # + self.k2 * self.link_pred(adj, s_l)
                # + self.k3 * self.entropy(s_l)
                )
