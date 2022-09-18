from torch import nn
import time
#from models import HGNN_conv
import torch.nn.functional as F
import math
import torch
import glog as log
from torch.nn.parameter import Parameter

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        t1 = time.time()
        x = x.matmul(self.weight)
        t1 = time.time() - t1
        if self.bias is not None:
            x = x + self.bias
        #x = G.matmul(x)
        t2 = time.time()
        log.info(x.shape)
        x = torch.sparse.mm(G,x)
        t2 = time.time() - t2
        return x,t1,t2


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        t1 = time.time();
        out1,t1_l1,t2_l1 = self.hgc1(x, G)
        t1 = time.time()-t1;
        
        x = F.relu(out1)
        x = F.dropout(x, self.dropout)
        
        t2 = time.time();
        x,t1_l2,t2_l2 = self.hgc2(x, G)
        t2 = time.time()-t2
        return x,t1,t2,t1_l1+t1_l2,t2_l1+t2_l2
