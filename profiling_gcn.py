from __future__ import division
from __future__ import print_function
import argparse
import time
import torch
import numpy as np
from numpy import argmax
import torch.nn.functional as F
from pygcn.gcnio.data import dataio
from pygcn.gcnio.util import utils
from pygcn.gcn6 import GCN
import scipy.sparse
import json
from sklearn.preprocessing import StandardScaler
import logging as log
import torch.optim as optim

log.basicConfig(filename='profiling-gcn.log', level=log.INFO);
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

smallgraphs = {'pubmed'}
largegraphs = {'flickr','ppi','amazon','reddit','yelp'}

parser = argparse.ArgumentParser("Graph to be processed ... ")
parser.add_argument('-g','--graph')
parser.add_argument('-k','--hidden', type=int)
parser.add_argument("-i",'--train-iters', dest='train_iters',
                    type=int, default=100 )
args = parser.parse_args()

SMALL = False
dataset = args.graph
if dataset in smallgraphs:
    SMALL = True
elif dataset in largegraphs:
    SMALL = False
else:
    print('Except input graph ... ')
    exit(1)
print(dataset)


# load original dataset (to get clean features and labels)
if SMALL:
    data = dataio.Dataset(root='/tmp/', name=dataset)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    log.info(type(adj))
    log.info(adj.shape)
    log.info(type(features))
    log.info(features.shape)
    log.info(type(labels))
    log.info(labels.shape[0])
    log.info(type(idx_train))
    log.info(idx_train.shape)
    log.info(type(idx_val))
    log.info(idx_val.shape)
    log.info(type(idx_test))
    log.info(idx_test.shape)
else:
    data_prefix = './dataset/'+dataset
    temp_data = load_data(data_prefix)
    data_list = data_prefix.split('/')
    dataset = data_list[-1]
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
    log.info(labels.shape[0])
    log.info(type(labels[0]))
    log.info(type(idx_train))
    log.info(idx_train.shape)
    log.info(type(idx_val))
    log.info(idx_val.shape)
    log.info(type(idx_test))
    log.info(idx_test.shape)


#print(labels[0])
#print(labels[1])
#print(labels[8])
print('labels = ',labels.max()+1)

model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max()+1, dataset=dataset, device=device)
  
optimizer = optim.Adam(model.parameters(),
                       lr=0.01, weight_decay=5e-4)


model = model.to(device)
TRAIN = 1
if TRAIN:
    model.fit(features, adj, labels, idx_train, train_iters=args.train_iters, verbose=True, name=dataset)
    #torch.save(model.state_dict(),'./model/gcn.pt')
TEST = 0
if TEST:
    model.load_state_dict(torch.load('./model/gcn.pt'))
    model.eval()
    model.test(idx_test)


