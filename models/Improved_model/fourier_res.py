import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size=1, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        if filter_size > 0:
            self.layer1 = nn.Linear(hidden_size, filter_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.layer2 = nn.Linear(filter_size, hidden_size)
        else:
            self.layer1 = nn.Linear(hidden_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.layer2 = nn.Linear(hidden_size, abs(filter_size))


        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class ReLuPhi(nn.Module):
    """ReLuPhi is the equivalent of resistors for each node in the graph."""

    def __init__(self, hidden_size, head_size=1):
        super(ReLuPhi, self).__init__()

        self.head_size = head_size
        self.phi = nn.Linear(hidden_size,hidden_size*head_size)
        initialize_weight(self.phi)

    def forward(self, x, f_norm, pos):
        """
            This function implements ReLU1(|f| - phi(x))
            x.shape:    batch, N, d
            f_norm:     batch, N, d
        """
        b, N, d = x.size()
        one = torch.ones(b,N,self.head_size,d).to(x.device)
        pos_enc = pos.unsqueeze(0).repeat(x.size(0),1,1)
        # For resistors we can pass in (x + pos) in stead of just x,
        # so that phi(.) also knows the frequency associated with a x.
        min_val = torch.minimum(one, f_norm.unsqueeze(2) - self.phi(x+pos_enc).view(b, N, self.head_size,-1)) #shape: b x N x H x d
        max_val = F.relu(min_val) # we can implement the minimum value as eps and not zero

        return max_val


class FourierResistor(nn.Module):
    """
        Uses a algorithm related to the Fourier Transform designed in my notes.
    """

    def __init__(self, hidden_size, head_size=4, dropout_rate=0.1):
        super(FourierResistor, self).__init__()

        self.linear_trans = nn.Linear(hidden_size, hidden_size, bias=False)
        self.relu_phi = ReLuPhi(hidden_size,head_size)
        self.resistors_drp_out = nn.Dropout(dropout_rate) #May remove this!!!!!!!!!!
        self.combine_heads = FeedForwardNetwork(head_size, -1)# nn.Linear(head_size,1, bias=False) #May need to be a full MLP
        self.sum_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, hidden_size, dropout_rate)

        initialize_weight(self.linear_trans)
        # initialize_weight(self.combine_heads)

    def eta(self, pos, conj=False):
        """
            This takes in the position encoding matrix and returns eta, which has been
            defined in my notes.
            eta (Greek symbol) is just exp(-i*theta) = cos(theta) - i*sin(theta).
            cos(theta) and sin(theta) are obtained from the columns of pos.
        """
############################################################################
        ### Consider changing the duplicate value used for the entries in the even position
        N, d = pos.size()
        e = torch.zeros(2,N,d).to(pos.device)
        e[0,:,0::2] = pos[:,1::2] #cosine
        e[0,:,1::2] = pos[:,1::2] #cosine, Duplicate in second position
        e[1,:,0::2] = pos[:,0::2] if conj else -pos[:,0::2] #sine
        e[1,:,1::2] = pos[:,0::2] if conj else -pos[:,0::2] #sine, Duplicate in second position

        return e #shape: 2 x N x d

    def fourier_trans(self, x, pos):
        """
            Implements the Fourier transform using the sinusoidal position encodings for exp(-i*theta).
            x.shape:    batch, N, d
            pos.shape:  N, d
        """
        N, _ = pos.size()
        A = ( torch.ones(N,N,dtype=torch.float) - torch.eye(N,dtype=torch.float) ).to(x.device)
        x_eta = x.mul(self.eta(pos).unsqueeze(1)) #shape: 2 x b x N x d
        f = A.matmul(x_eta) #shape: 2 x b x N x d
        return f, A


    def forward(self, x, pos, N=None):
        """
            x.shape:    batch, N, d
            pos.shape:  N, d
        """
        if len(pos.size())==3 :
            pos = pos[0]
        # Need to perform a linear transformation of the
        # input x before passing for a Fourier transform
        x = self.linear_trans(x)
        f, A = self.fourier_trans(x, pos) #shape: 2 x b x N x d
        f_norm = torch.norm(f, dim=0) #shape: b x N x d

        resistors = self.resistors_drp_out(self.relu_phi(x, f_norm, pos)) #shape: b x N x H x d
        conjugate = self.eta(pos, conj=True) #shape: 2 x N x d

        # divisor for inverse transform (we can also learn this scale)
        scale = (1./pos.size(0))
        # Real part of f*eta_conjugate
        f_eta_conj = f[0].mul(conjugate[0]) - f[1].mul(conjugate[1]) #shape: b x N x d
        # inverse transform
        res_f_eta = resistors.mul(f_eta_conj.unsqueeze(2)) #shape: b x N x H x d
        # Make this b x H x d x N then right multiply by N x N
        inverse_transform = scale*res_f_eta.permute(0,2,3,1).matmul(A) #shape: b x H x d x N
        # reshape to b x N x H x d
        inverse_transform = inverse_transform.permute(0,3,1,2) #shape: b x N x H x d
        modulated_signal = x + self.combine_heads(inverse_transform.transpose(-1,-2)).squeeze(-1) #shape: b x N x d
        modulated_signal = self.sum_norm(modulated_signal) #layer norm

        return self.ffn(modulated_signal)
