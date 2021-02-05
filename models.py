import torch.nn as nn
import torch.nn.functional as F
from layers import LayerConvol, AttenMlp
import torch
import math
from torch.nn.parameter import Parameter


class nrecGNN(nn.Module):
    def __init__(self, nfeat, n_hops, act, hidden_state,
                 share_attn, nclass, dropnode=0.0):
        super(nrecGNN, self).__init__()
        self.input_dim = nfeat
        self.n_hops = n_hops
        self.nclass = nclass
        self.share_attn = share_attn
        self.hidden_state = hidden_state
        self.dropnode = dropnode
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()
        self.dropout = [0.6 for i in range(n_hops)]
        self.__init()

    def __init(self):
        if self.share_attn:
            self.gc1 = LayerConvol(self.input_dim, self.input_dim)
        else:
            self.gc1 = [LayerConvol(self.input_dim, self.input_dim) for i in range(self.n_hops)]
        self.atten = AttenMlp(self.input_dim)
        self.linear = nn.Linear(self.input_dim, self.hidden_state, bias=True)
        self.linear2 = nn.Linear(self.hidden_state, self.nclass, bias=True)

    def __dropout_x(self, x, idx):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        if len(self.dropout) == 1:
            dropt = self.dropout[-1]
        else:
            dropt = self.dropout[idx]

        random_index = torch.rand(len(values)) + dropt
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def forward(self, x, adj, idx):
        seq_emb = []
        seq_emb.append(F.normalize(x[idx], p=2, dim=1))
        xx_anchor = seq_emb[0]
        if self.training:
            adj = [self.__dropout_x(adj_, i) for i, adj_ in enumerate(adj)]

        for i, adj_ in enumerate(adj):
            if self.share_attn:
                x_ = self.gc1(xx_anchor, x, adj_)
            else:
                x_ = self.gc1[i](xx_anchor, x, adj_)

            seq_emb.append(F.normalize(x_, p=2, dim=1))

        seq_emb = torch.stack(seq_emb, dim=1)
        output = self.atten(xx_anchor, seq_emb, seq_emb)
        output = F.dropout(output, self.dropnode, training=self.training)
        output = self.linear(output)
        output = self.act(output)
        output = self.linear2(output)
        return F.log_softmax(output, dim=1)
