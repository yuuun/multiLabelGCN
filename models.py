import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        
        #x = nn.Sigmoid()(x) 
        '''
        if use nn.Sigmoid(), then it does not train well --> uses nn.sigmoid() function on testing
        '''
        
        return x
