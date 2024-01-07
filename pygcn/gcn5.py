import torch.nn as nn
import torch.nn.functional as F
import math
import time
import torch
import torch.optim as optim
import logging as log
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from pygcn.gcnio.util import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from pygcn.writecsv import save 

from pygcn.perf import dmk
import ctypes

cuspmmLib = ctypes.cdll.LoadLibrary('./cuspmm.so')


# A(XW)
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=True, name='dataset', layer='layer0'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.layer = layer
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.timers = dmk.Timers()
        print(f"data: {name} -> layer = {layer}: A(XW) "
              f"{in_features} ✕ {out_features}  bias {with_bias}" )

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_timing(self):
        self.timers.reset()

    def forward(self, input, adj):

        if input.data.is_sparse:
            if False:
                print("@59...",input.is_sparse)
                print("@60...",input.layout==torch.sparse_coo)
                print("@61...",input.coalesce().indices().shape[0])
                print("@62...",input.coalesce().indices().shape[1])
                print("@63...",input.coalesce().values().shape[0])
                print("@64...",type(input))
                print("@65...",input.to_sparse_csr().crow_indices().shape)
                print("@66...",input.to_sparse_csr().col_indices().shape)
                print("@67...",input.to_sparse_csr().values().shape)
            if False:
                input_csr = input.to_sparse_csr()
                m = input_csr.crow_indices().shape[0]-1 
                n = input.to_sparse_csc().ccol_indices().shape[0]-1
                nnz = input_csr.values().shape[0]
                dim = self.weight.shape[1]
                print("m=",m,",n=",n,",nnz=",nnz,",dim=",dim)
                support = torch.zeros((m,dim), device='cuda')
                with self.timers.hc.xw:
                    cuspmmLib.cuspmm(ctypes.c_void_p(input_csr.crow_indices().data_ptr()), 
                               ctypes.c_void_p(input_csr.col_indices().data_ptr()),
                               ctypes.c_void_p(input_csr.values().data_ptr()),
                               ctypes.c_void_p(self.weight.data_ptr()),
                               ctypes.c_void_p(support.data_ptr()),
                               m,n,nnz,dim)
            else:
                with self.timers.hc.xw:
                    support = torch.spmm(input, self.weight)
        else:
            with self.timers.hc.xw:
                support = torch.mm(input, self.weight)
        with self.timers.hc.af:
          output = torch.spmm(adj, support)
        if self.bias is not None:
            with self.timers.hc.bi: output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# (AX)W
class GraphConvolution2(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=True, name='dataset', layer='layer0'):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.layer = layer
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.timers = dmk.Timers()
        print(f"data: {name} -> layer = {layer}: (AX)W "
              f"{in_features} ✕ {out_features}  bias {with_bias}" )

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_timing(self):
        self.timers.reset()

    def forward(self, input, adj):
        with self.timers.hc.af: support = torch.spmm(adj, input)
        with self.timers.hc.xw: output = torch.mm(support, self.weight)
        if self.bias is not None:
            with self.timers.hc.bi: output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dataset,  
            dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.dataname = dataset
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias, name=dataset, layer='layer1')
        if dataset=='pubmed' or dataset=='flickr': 
            self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias, name=dataset, layer='layer2')
        else:
            self.gc2 = GraphConvolution2(nhid, nclass, with_bias=with_bias, name=dataset, layer='layer2')
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.dur_fwd = dmk.Timer()

    def reset_timing(self):
        self.dur_fwd.reset();
        for gc in [self.gc1,self.gc2]: gc.reset_timing();
        
    def forward(self, x, adj, name='dataset'):
        '''
            adj: normalized adjacency matrix
        '''

        with self.dur_fwd:
          x = self.gc1(x, adj);
          if self.with_relu: x = F.relu(x)
          x = F.dropout(x, self.dropout, training=self.training)
          x = self.gc2(x, adj)
          x = F.log_softmax(x, dim=1)
        return x

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, name='dataset'):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            print("Transform data to GPU device ...")
            adj, features, labels = utils.to_tensor(adj, features, labels, device=self.device)
            #log.info(adj.dtype)
            #log.info(features.dtype)
            #return ;
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        print(f"Graph size: {features.shape[0]} vertices "
              f"Input Features: {features.shape[1]} "
              f"is_sp {features.data.is_sparse}" )

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj
        
        if False and utils.is_sparse_tensor(adj_norm):
            save.write(adj_norm, name)
        
        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels
        
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose, name)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose, name)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose, name)
        
        # print('Forward time: {:.4f}s'.format(self.t_fp))
        # print(' Layer1 time: {:.4f}s'.format(self.t_fp_l1))
        # print('     Layer1 spmm time: {:.4f}s'.format(self.t_fp_spmm_l1))
        # print('     Layer1 mm time: {:.4f}s'.format(self.t_fp_mm_l1))
        # print(' Layer2 time: {:.4f}s'.format(self.t_fp_l2))
        # print('     Layer2 spmm time: {:.4f}s'.format(self.t_fp_spmm_l2))
        # print('     Layer2 mm time: {:.4f}s'.format(self.t_fp_mm_l2))
        # print('Backward time: {:.4f}s'.format(self.t_bp))

        print(f"Forward time: {self.dur_fwd.s():.4f} s "
              f"for {self.dur_fwd.n_calls} calls.");

        for gc in [self.gc1,self.gc2]:
            print(f"{gc.layer} xw: {gc.timers.h.xw.avms():6.4f} ms  "
              f" cu {gc.timers.c.xw.avms():6.4f} ms "
              f" af: {gc.timers.h.af.avms():7.4f} ms "
              f" cu {gc.timers.c.af.avms():7.4f} ms "
              f" bi: {gc.timers.h.bi.avms():7.4f} ms "
              f" cu {gc.timers.c.bi.avms():7.4f} ms" );

    def _train_without_val(self, labels, idx_train, train_iters, verbose, name):
        print('_train_without_val')
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        ti = dmk.Timers()
        warmup_upto = 10 if train_iters >= 20 else 1 if train_iters == 1 else 0

        for i in range(train_iters):
            optimizer.zero_grad()
            with ti.h.fwd:
                output = self.forward(self.features, self.adj_norm, name)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            with ti.h.bwd:
                loss_train.backward()
            optimizer.step()
            if verbose and ( i % 10 == 0 or i == warmup_upto ):
                print(f'Epoch {i:3d}, training loss: {loss_train.item():.6f}'
                      f'  Fwd: {ti.h.fwd.avms():.3f} ms/iter'
                      f'  Bwd: {ti.h.bwd.avms():.3f} ms/iter',
                      end='')
                ti.reset()
                if i == warmup_upto:
                    self.reset_timing();
                    print(" Warmup ending.",end='')
                print("");
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose, name):
        print('_train_with_val')
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm, name)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        print('_train_with_early_stopping')
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        output, t_l1, t1_l1, t_relu, t_l2, t1_l2, t2_l2 = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test

    def _set_parameters():
        # TODO
        pass

    def predict(self, features=None, adj=None):
        '''By default, inputs are unnormalized data'''

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)

