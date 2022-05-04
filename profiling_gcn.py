from __future__ import division
from __future__ import print_function

import time
import torch
import numpy as np
from numpy import argmax
import torch.nn.functional as F
from pygcn.gcnio.data import dataio
from pygcn.gcnio.util import utils
from pygcn.gcn1 import GCN
import scipy.sparse
import json
from sklearn.preprocessing import StandardScaler
import glog as log
import torch.optim as optim

cuda = torch.cuda.is_available()
print('cuda: %s' % cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

def load_data(prefix, normalize=True):
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(prefix))
    adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(prefix))
    role = json.load(open('./{}/role.json'.format(prefix)))
    feats = np.load('./{}/feats.npy'.format(prefix))
    class_map = json.load(open('./{}/class_map.json'.format(prefix)))
    class_map = {int(k):v for k,v in class_map.items()}
    assert len(class_map) == feats.shape[0]
    # ---- normalize feats ----
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------
    return adj_full, adj_train, feats, class_map, role


def process_graph_data(adj_full, adj_train, feats, class_map, role, name):
    """
    setup vertex property map for output classes, train/val/test masks, and feats
    INPUT:
        G           graph-tool graph, full graph including training,val,testing
        feats       ndarray of shape |V|xf
        class_map   dictionary {vertex_id: class_id}
        val_nodes   index of validation nodes
        test_nodes  index of testing nodes
    OUTPUT:
        G           graph-tool graph unchanged
        role        array of size |V|, indicating 'train'/'val'/'test'
        class_arr   array of |V|x|C|, converted by class_map
        feats       array of features unchanged
    """
    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0],list):
        print("labels are list")
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, 1))
        p = 0;
        for k,v in class_map.items():
            class_arr[p] = argmax(v)
            p = p+1
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, 1))
        for k,v in class_map.items():
            class_arr[k] = v
    if name=='flickr' or name=='reddit' or name=='ppi' or name=='amazon' or name=='yelp':
        class_arr = np.squeeze(class_arr.astype(int))

    return adj_full, adj_train, feats, class_arr, role


# make sure you use the same data splits as you generated attacks
seed = 15
np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

# load original dataset (to get clean features and labels)
SMALL = False
if SMALL:
    dataset = 'polblogs'
    data = dataio.Dataset(root='/tmp/', name=dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    log.info(type(adj))
    log.info(adj.shape)
    log.info(type(features))
    log.info(features.shape)
    log.info(type(labels))
    log.info(labels.shape)
    log.info(type(idx_train))
    log.info(idx_train.shape)
    log.info(type(idx_val))
    log.info(idx_val.shape)
    log.info(type(idx_test))
    log.info(idx_test.shape)
else:
    data_prefix = './dataset/amazon'
    temp_data = load_data(data_prefix)
    data_list = data_prefix.split('/')
    print(data_list[-1])
    train_data = process_graph_data(*temp_data,data_list[-1])
    adj,adj_train,features,labels,role = train_data
    features = scipy.sparse.csr_matrix(features)
    idx_train = np.array(role['tr'])
    idx_val = np.array(role['va'])
    idx_test = np.array(role['te'])
    log.info(type(adj))
    log.info(adj.shape)
    log.info(type(adj_train))
    log.info(adj_train.shape)
    log.info(type(features))
    log.info(features.shape)
    log.info(type(labels))
    log.info(labels.shape)
    log.info(type(labels[0]))
    log.info(type(idx_train))
    log.info(idx_train.shape)
    log.info(type(idx_val))
    log.info(idx_val.shape)
    log.info(type(idx_test))
    log.info(idx_test.shape)


print(labels[0])
print(labels[1])
print(labels[8])
print(labels.max())

model = GCN(nfeat=features.shape[1], nhid=32, nclass=labels.max()+1, device=device)
  
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)


model = model.to(device)
TRAIN = 1
if TRAIN:
    model.fit(features, adj, labels, idx_train, train_iters=1, verbose=True, name='amazon')
    torch.save(model.state_dict(),'./model/gcn.pt')
TEST = 1
if TEST:
    model.load_state_dict(torch.load('./model/gcn.pt'))
    model.eval()
    model.test(idx_test)


