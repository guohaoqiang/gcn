from torch import nn
import time
from models import HGNN_conv
import torch.nn.functional as F


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
