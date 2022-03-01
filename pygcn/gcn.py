import torch.nn as nn
import torch.nn.functional as F
import math
import time
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from pygcn.gcnio.util import utils
from copy import deepcopy
from sklearn.metrics import f1_score

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, name='dataset', layer='layer0'):
        t1 = 0
        if input.data.is_sparse:
            #print(name+' : '+layer+' : '+'spmm')
            t1 = time.time()
            support = torch.spmm(input, self.weight)
            t1 = time.time()-t1
        else:
            #print(name+' : '+layer+' : '+'mm')
            t1 = time.time()
            support = torch.mm(input, self.weight)
            t1 = time.time()-t1
        t2 = time.time()    
        output = torch.spmm(adj, support)
        t2 = time.time()-t2
        if self.bias is not None:
            return output + self.bias, t1, t2
        else:
            return output, t1, t2

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)
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
        
        self.t_fp = 0
        self.t_bp = 0
        self.t_fp_l1 = 0
        self.t_fp_t1_l1 = 0
        self.t_fp_t2_l1 = 0
        self.t_fp_l2 = 0
        self.t_fp_t1_l2 = 0
        self.t_fp_t2_l2 = 0

    def forward(self, x, adj, name='dataset'):
        '''
            adj: normalized adjacency matrix
        '''
        t_l1 = 0   # time cost in the first layer
        t_l2 = 0   # time cost in the second layer
        if self.with_relu:
            t_l1 = time.time()
            x, t1_l1, t2_l1 = self.gc1(x, adj, name, layer='Layer1')
            x = F.relu(x)
            t_l1 = time.time() - t_l1;
        else:
            t_l1 = time.time()
            x, t1_l1, t2_l1 = self.gc1(x, adj, name, layer='Layer1')
            t_l1 = time.time() - t_l1;
        
        t_l2 = time.time()
        x = F.dropout(x, self.dropout, training=self.training)
        x, t1_l2, t2_l2  = self.gc2(x, adj, name, layer='Layer2')
        t_l2 = time.time() - t_l2;
        return F.log_softmax(x, dim=1), t_l1, t1_l1, t2_l1, t_l2, t1_l2, t2_l2

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
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

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
        
        print('Forward time: {:.4f}s'.format(self.t_fp))
        print('Layer1 time: {:.4f}s'.format(self.t_fp_l1))
        print('Layer1 XW time: {:.4f}s'.format(self.t_fp_t1_l1))
        print('Layer1 A(XW) time: {:.4f}s'.format(self.t_fp_t2_l1))
        print('Layer2 time: {:.4f}s'.format(self.t_fp_l2))
        print('Layer2 XW time: {:.4f}s'.format(self.t_fp_t1_l2))
        print('Layer2 A(XW) time: {:.4f}s'.format(self.t_fp_t2_l2))
        print('Backward time: {:.4f}s'.format(self.t_bp))

    def _train_without_val(self, labels, idx_train, train_iters, verbose, name):
        print('_train_without_val')
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        for i in range(train_iters):
            optimizer.zero_grad()
            temp1 = time.time()
            output, t_l1, t1_l1, t2_l1, t_l2, t1_l2, t2_l2 = self.forward(self.features, self.adj_norm, name)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            self.t_fp += (time.time()-temp1)
            
            temp2 = time.time()
            loss_train.backward()
            optimizer.step()
            self.t_bp += (time.time()-temp2)
            
            self.t_fp_l1 += t_l1
            self.t_fp_t1_l1 += t1_l1 
            self.t_fp_t2_l1 += t2_l1
            self.t_fp_l2 += t_l2
            self.t_fp_t1_l2 += t1_l2
            self.t_fp_t2_l2 += t2_l2
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output, t_l1, t1_l1, t2_l1, t_l2, t1_l2, t2_l2 = self.forward(self.features, self.adj_norm, name)
        self.t_fp_l1 += t_l1
        self.t_fp_t1_l1 += t1_l1 
        self.t_fp_t2_l1 += t2_l1
        self.t_fp_l2 += t_l2
        self.t_fp_t1_l2 += t1_l2
        self.t_fp_t2_l2 += t2_l2
        
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
        output, t_l1, t1_l1, t2_l1, t_l2, t1_l2, t2_l2 = self.predict()
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

