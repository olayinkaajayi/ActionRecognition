import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from zero_one import Inverse_line, Inverse_log_line, Exp_zero_one, Sig_Linear, Sig_Linear2, Sig_Linear3

def toNumpy(v,device='cuda'):
    """This function converts the entry 'v' back to a numpy (float) data type"""
    if device == 'cuda':
        return v.detach().cpu().numpy()
    return v.detach().numpy()


class Position_encode(torch.nn.Module):
    """This helps learn the positional encoding for our graph."""

    def __init__(self, A=None, d=None, temperature=1.0, N=None):
        super(Position_encode, self).__init__()

        self.temperature = temperature
        self.base_temperature = 1.0 #set as 1 till I know what to do with it

        self.sigmoid = Sig_Linear2()

        if A is not None:
            N = A.shape[0]
        self.N = N

        if d is None:
            d = int(np.ceil(np.log2(N)))
            print(f"Dimension d={d}")

        self.d = d

        self.P = nn.Parameter(torch.randn(N,d))
        self.W_d = nn.Parameter(torch.randn(d))

        if A is not None:
            print("Setting up Graph, deg vec, Edge Idx & Deg dist Idx...")
            self.G = nx.from_numpy_matrix(A)
            self.deg_vec = self.degree_vec(A)
            self.edg_idx = self.edge_index(A)
            self.deg_dist_edg_idx = self.deg_dist_edge_index(A)


    def deg_dist_edge_index(self, adj):
        """
            Returns an edge list for the degree distribution,
            that is, nodes with the same edges have the same degree.
        """
        degree = self.deg_vec

        edge_list = []
        for i in range(self.N):
            # using this makes it faster
            idx = torch.where(degree==degree[i])[0]
            if len(idx) != 0:
                for j in idx:
                    if (i != j):
                        edge_list.append([i,j])
            # for j in range(self.N):
            #     if (degree[i] == degree[j]) and (i != j):
            #         edge_list.append([i,j])

        return torch.tensor(edge_list).t() #shape: 2 x |Q|, |Q|: num of edges in the distribution graph


    def edge_index(self, adj):
        """Computes the edge index from the adjacency matrix"""
        adj = torch.from_numpy(adj)
        return adj.nonzero().t().contiguous() #shape: 2 x |E|, |E|: num of edges


    def degree_vec(self, adj):
        """Computes the degree of each node"""
        adj = torch.from_numpy(adj)
        return torch.sum(adj,dim=0) #shape: N x 1


    def neighbours(self, deg_dist=False):
        """Returns the adjacency list"""
        if deg_dist:
            edge_idx = self.deg_dist_edg_idx
        else:
            edge_idx = self.edg_idx

        #negative neighbours is wrong
        adj_list = [ edge_idx[ 1, torch.where( edge_idx[0,:] == i )[0] ] for i in range(self.N) ]
        # This cannot determine negative neighbours accurately. Would need to use set difference
        # neg_adj_list = [ edge_idx[ 1, torch.where( edge_idx[0,:] != i )[0] ] for i in range(self.N) ]

        return adj_list


    def hamming_dist(self, zi, zp):
        """Computes hamming distance using absolute value"""
        return (zi - zp).abs().sum()


    def degree_loss(self, Z):
        """Difference between degree vectors"""
        deg_prime = torch.matmul(Z, self.W_d) #shape: N x 1
        return torch.norm(deg_prime - self.deg_vec.to(deg_prime.device))


    def return_k_hops(self, i, max_depth=5):
        """
            This returns the k-hop neighbours for the negative examples. We are interested
            in k>1 hops. We need the value of k as weights when computing softmax of negative examples.

            We choose not to search beyond a maximum depth of 5, as most
            nodes far away would not constitute much problem.
        """
        length = nx.single_source_shortest_path_length(self.G,source=i,cutoff=max_depth)
        # length is a dictionary that gives all the nodes reachable from i and thier distance.
        return list(length.keys()) #We return just the keys (nodes far from i [including i])


    def get_faraway_neg_neighbours(self, node_i, max_sample=100, max_depth=5):
        """
            We want the negative neighbours to be those nodes that are
            reasonably far from it.
            Then we sample from this list of negative neighbours.
        """
        if max_sample > self.N:
            # If we have a small graph, we default to the normal model
            max_depth = None
            hops = self.return_k_hops(node_i, max_depth)
            hops.remove(node_i)
            return hops

        close_neigh = self.return_k_hops(node_i, max_depth)
        to_compare = set( close_neigh )
        neg_neigh = set(range(self.N)).difference( to_compare )

        # Now we sample from the negative neighbour
        neg_neigh = self.should_sample(neg_neigh, max_sample = 100)

        return neg_neigh


    def should_sample(self, neg_neigh, max_sample=100):
        """This decides if we should sample or not."""
        if len(neg_neigh) > max_sample:
            neg_neigh = random.sample(neg_neigh, max_sample)

        return neg_neigh


    def contrast_adj(self, Z, nodes, deg_dist=False):
        """
            Learn embeddings based on hamming distance and supervised contrastive loss.
            The purpose of this is to use the adjacency graph to learn the embeddings
            of vectors so they are close together as needed.
        """
        total = 0
        pos_neigh = self.neighbours(deg_dist)
        # for i in range(self.N):
        for i in nodes:

            sum_pos = 0
            for p in pos_neigh[i]:
                num = self.hamming_dist(Z[i], Z[p]) / self.temperature
                sum_pos += num

            sum_neg = 0
            if not deg_dist:
                hops = self.get_faraway_neg_neighbours(i, max_sample=100, max_depth=5)
                for a in hops:
                    den = self.hamming_dist(Z[i], Z[a]) / self.temperature
                    # forgot to take the exponential of the hamming distance
                    # fixed on 8th Feb, 2023.
                    sum_neg += torch.exp(den)
            else:
                to_compare = set( toNumpy(pos_neigh[i]).tolist() )
                to_compare.add(i)
                neg_neigh = set(range(self.N)).difference( to_compare )

                # Now we sample from the negative neighbour
                neg_neigh = self.should_sample(neg_neigh, max_sample = 100)
                for a in neg_neigh:
                    den = self.hamming_dist(Z[i], Z[a]) / self.temperature
                    # forgot to take the exponential of the hamming distance
                    # fixed on 8th Feb, 2023.
                    sum_neg += torch.exp(den)


            log_den = torch.log(sum_neg)

            len_pos_neigh = len(pos_neigh[i])
            if (len_pos_neigh == 0):
                len_pos_neigh = 1.0 #To avoide dividing by zero. This will not contribute to the result

            total += (- sum_pos / len_pos_neigh) + log_den

        return total/self.N


    def contrast_dist(self, Z, nodes, deg_dist=True):
        """
            Learn embeddings based on hamming distance and supervised contrastive loss.
            The purpose of this is to use the degree distribution graph to learn the embeddings
            of vectors so they are relatively close together as needed.
        """
        return self.contrast_adj(Z, nodes, deg_dist=deg_dist)


    def forward(self, selected_nodes=None, test=False, deg=False):
        """Implements the proposed algorithm"""
        if test:
            if deg:
                # Z = torch.sigmoid(self.P)
                Z = self.sigmoid(self.P)
                return Z, torch.matmul(Z, self.W_d), self.deg_vec
            # return torch.sigmoid(self.P), None, None
            return self.sigmoid(self.P), None, None

        # Z = torch.sigmoid(self.P)
        Z = self.sigmoid(self.P)
        # print("Z=\n",torch.round(Z))
        # exit()

        nodes = selected_nodes.cpu().numpy()

        L_deg = self.degree_loss(Z)

        L_adj = self.contrast_adj(Z, nodes)

        L_deg_dist = self.contrast_dist(Z, nodes)

        return L_deg, L_adj, L_deg_dist
