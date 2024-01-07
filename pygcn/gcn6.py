import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import torch
import torch.optim as optim
import logging as log
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from pygcn.gcnio.util import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from pygcn.writecsv import save 

from pygcn.perf import dmk
import ctypes

cuspmmLib = ctypes.cdll.LoadLibrary('./cuspmm.so')
flexspmmLib = ctypes.cdll.LoadLibrary('./flexspmm.so')
renumberLib = ctypes.cdll.LoadLibrary('./renumber.so')
tileLib = ctypes.cdll.LoadLibrary('./tile.so')
permutateLib = ctypes.cdll.LoadLibrary('./permutate.so')

class flexspmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, seg_rowPtr, segNzCV, segVoMap, m, n, n_segs, grouped_tailSeg, next_seg, input, output):
        flexspmmLib.flexspmm(ctypes.c_void_p(seg_rowPtr.data_ptr()),
                            ctypes.c_void_p(segNzCV.data_ptr()),
                            ctypes.c_void_p(segVoMap.data_ptr()),
                            ctypes.c_void_p(grouped_tailSeg.data_ptr()),
                            ctypes.c_void_p(next_seg.data_ptr()),
                            m, n, input.shape[1], n_segs,
                            ctypes.c_void_p(input.data_ptr()),
                            ctypes.c_void_p(output.data_ptr()))
        ctx.backward_flex = seg_rowPtr, segNzCV, segVoMap, m, n, n_segs, grouped_tailSeg, next_seg
        return output
    @staticmethod
    def backward(ctx, grad_out):
        seg_rowPtr, segNzCV, segVoMap, m, n, n_segs, grouped_tailSeg, next_seg = ctx.backward_flex 
        grad_x = torch.zeros( (m, grad_out.shape[1]), device="cuda" ) 
        flexspmmLib.flexspmm(ctypes.c_void_p(seg_rowPtr.data_ptr()),
                            ctypes.c_void_p(segNzCV.data_ptr()),
                            ctypes.c_void_p(segVoMap.data_ptr()),
                            ctypes.c_void_p(grouped_tailSeg.data_ptr()),
                            ctypes.c_void_p(next_seg.data_ptr()),
                            m, n, grad_out.shape[1], n_segs,
                            ctypes.c_void_p(grad_out.data_ptr()),
                            ctypes.c_void_p(grad_x.data_ptr()))
        grad_edge_weight = None
        return None,None,None,None,grad_x,grad_edge_weight,None

        
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

    def forward(self, input, seg_rowPtr, segNzCV, segVoMap, m, n, n_segs, grouped_tailSeg, next_seg):

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
                x_m = input_csr.crow_indices().shape[0]-1 
                x_n = input.to_sparse_csc().ccol_indices().shape[0]-1
                x_nnz = input_csr.values().shape[0]
                x_dim = self.weight.shape[1]
                print("m=",x_m,",n=",x_n,",nnz=",x_nnz,",dim=",x_dim)
                support = torch.zeros((x_m,x_dim), device='cuda')
                with self.timers.hc.xw:
                    cuspmmLib.cuspmm(ctypes.c_void_p(input_csr.crow_indices().data_ptr()), 
                               ctypes.c_void_p(input_csr.col_indices().data_ptr()),
                               ctypes.c_void_p(input_csr.values().data_ptr()),
                               ctypes.c_void_p(self.weight.data_ptr()),
                               ctypes.c_void_p(support.data_ptr()),
                               x_m,x_n,x_nnz,x_dim)
            else:
                with self.timers.hc.xw:
                    support = torch.spmm(input, self.weight)
        else:
            with self.timers.hc.xw:
                support = torch.mm(input, self.weight)
        
        # directly create tensor on GPU
        output = torch.zeros((m,self.weight.shape[1]), device='cuda')
        with self.timers.hc.af:
          #output = torch.spmm(adj, support)
          flexspmm.apply(seg_rowPtr, segNzCV, segVoMap, 
                        m, n, n_segs.item(), 
                        grouped_tailSeg, next_seg,
                        support, output)
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
        
    def forward(self, x, seg_rowPtr, segNzCV, m, n, n_segs, segVoMap, grouped_tailSeg, next_seg, name='dataset'):
        '''
            adj: normalized adjacency matrix
        '''

        with self.dur_fwd:
          x = self.gc1(x, seg_rowPtr, segNzCV, segVoMap, m, n, n_segs, grouped_tailSeg, next_seg)
          if self.with_relu: x = F.relu(x)
          x = F.dropout(x, self.dropout, training=self.training)
          x = self.gc2(x, seg_rowPtr, segNzCV, segVoMap, m, n, n_segs, grouped_tailSeg, next_seg)
          x = F.log_softmax(x, dim=1)
        return x

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def permutateIdx(self, idx, vo_mp):
        temp = {}
        for i in range(vo_mp.shape[0]):
            key = vo_mp[i].item()
            temp[key] = i
        return [temp[i] for i in idx]

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, name='dataset'):
        '''
            train the gcn model, when idx_val is not None, pick the best model
            according to the validation loss
        '''
        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            #print("Transform data to GPU device ...")
            #adj, features, labels = utils.to_tensor(adj, features, labels, device=self.device)
            features, labels = utils.to_tensor2(features, labels)
            #log.info(adj.dtype)
            #log.info(features.dtype)
            #return ;
        
        if normalize:
            if sp.issparse(adj):
                adj_norm = utils.normalize_adj_tensor2(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor2(adj)
        else:
            adj_norm = adj

        print((adj_norm.to_dense().transpose(1,0)==adj_norm.to_dense()).all())
        
        if False and utils.is_sparse_tensor(adj_norm):
            save.write(adj_norm, name)
        # -- now, adj is a sparse tensor in COO. features and labels are tensors ----
        
        # step1: renumber (CPU)
        # step2: tiling + reassignment (CPU)
        # step3: transform data to GPU
        # step4: permutate feature matrix ( X(GPU) & labels(CPU) ).  
        #        The results are permuated as well. 
        #        The outputs of Layer1 can be used directly in Layer2
        #        The outputs of Layer2 can be used directly by loss computation

        ## step1: renumber (CPU)
        adj_csr = adj_norm.to_sparse_csr()
        self.m = adj_csr.crow_indices().shape[0]-1
        self.n = adj_csr.crow_indices().shape[0]-1
        self.nnz = adj_csr.values().shape[0]
        print("m=",self.m,",n=",self.n,",nnz=",self.nnz)
        vo_mp = torch.empty(self.m,dtype=int)
        adj_rowPtr = adj_csr.crow_indices().to(torch.int32)
        adj_col = adj_csr.col_indices().to(torch.int32)
        adj_values = adj_csr.values()
        vo_mp = vo_mp.to(torch.int32)
        renumberLib.dfs(ctypes.c_void_p(adj_rowPtr.data_ptr()), 
                        ctypes.c_void_p(adj_col.data_ptr()),
                        ctypes.c_void_p(adj_values.data_ptr()),
                        ctypes.c_void_p(vo_mp.data_ptr()),
                        self.m,self.n,self.nnz)
        print("renumber complete") 
        ## step2: tiling + reassignment (CPU)
        seg_rowPtr = torch.empty(self.nnz, dtype=torch.int32)
        segNzCV = torch.empty(2*self.nnz, dtype=torch.float32)
        segVoMap = torch.empty(self.nnz, dtype=torch.int32)
        grouped_tailSeg = torch.empty(256, dtype=torch.int32) # now we assume #SM is 256
        next_seg = torch.empty(256, dtype=torch.int32) 
        n_segs = torch.zeros(1, dtype=torch.int32)
        tm = 8
        tileLib.csr2tile(ctypes.c_void_p(adj_rowPtr.data_ptr()), 
                         ctypes.c_void_p(adj_col.data_ptr()),
                         ctypes.c_void_p(adj_values.data_ptr()),
                         self.m, self.n, self.nnz,
                         ctypes.c_void_p(vo_mp.data_ptr()),
                         ctypes.c_void_p(segVoMap.data_ptr()),
                         ctypes.c_void_p(seg_rowPtr.data_ptr()),
                         ctypes.c_void_p(segNzCV.data_ptr()),
                         ctypes.c_void_p(grouped_tailSeg.data_ptr()),
                         ctypes.c_void_p(next_seg.data_ptr()),
                         tm,
                         ctypes.c_void_p(n_segs.data_ptr())) 
        seg_rowPtr.resize_((tm+1)*n_segs[0]) 
        segVoMap.resize_(tm*n_segs[0]) 
        print("tiling complete") 
        ## step3: transform data to GPU
        self.k = features.shape[1]
        self.features = features.to_dense().to(self.device)
        #print("k = ",k,"f[0] = ",features[0])
        self.vo_mp = vo_mp.to(self.device)
        self.seg_rowPtr = seg_rowPtr.to(self.device)
        self.segNzCV = segNzCV.to(self.device)
        self.segVoMap = segVoMap.to(self.device)
        self.grouped_tailSeg = grouped_tailSeg.to(self.device)
        self.next_seg = next_seg.to(self.device)
        self.n_segs = n_segs[0]
        print("transform complete") 

        ## step4: permutate feature matrix ( X(GPU) & labels(CPU) ).  
        #        The results are permuated as well. 
        #        The outputs of Layer1 can be used directly in Layer2
        #        The outputs of Layer2 can be used directly by loss computation
        self.labels = labels.to(torch.int32).to(self.device)
        permutateLib.permutate(ctypes.c_void_p(self.features.data_ptr()),
                               ctypes.c_void_p(self.vo_mp.data_ptr()),
                               ctypes.c_void_p(self.labels.data_ptr()),
                               self.m, self.n, self.k)
        idx_train = self.permutateIdx(idx_train,self.vo_mp)
        print("permutate complete")
        print(f"Graph size: {features.shape[0]} vertices "
              f"Input Features: {features.shape[1]} "
              f"is_sp {features.data.is_sparse}" )
       
        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose, name)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(self.labels, idx_train, idx_val, train_iters, patience, verbose, name)
            else:
                self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose, name)
        
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
                output = self.forward(self.features, self.seg_rowPtr, self.segNzCV, 
                                      self.m, self.n, self.n_segs, 
                                      self.segVoMap, self.grouped_tailSeg, self.next_seg, name)
            labels = labels.to(torch.int64)
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

