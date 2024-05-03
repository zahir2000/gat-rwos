import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1))

        attention = torch.softmax(e.view(N, N), dim=1)
        attention = torch.mul(attention, adj)  # Apply the adjacency matrix to mask invalid edges

        h_prime = torch.matmul(attention, h)

        return h_prime, attention

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATLayer(nfeat, nhid)
        self.gat2 = GATLayer(nhid, nclass)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x, attn1 = self.gat1(x, adj)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x, attn2 = self.gat2(x, adj)
        return F.log_softmax(x, dim=1), attn1, attn2
