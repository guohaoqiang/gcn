import glog as log
import csv
import numpy as np
from scipy.sparse import csr_matrix

def write(adj,name):
    log.info(name)
    log.info(type(adj))
    log.info(adj.shape)
    
    adj_coo = adj.to("cpu")#.tolist()
    row = np.array(adj_coo.coalesce().indices().tolist()[0])
    col = np.array(adj_coo.coalesce().indices().tolist()[1])
    data = np.array(adj_coo.coalesce().values().tolist())
    
    log.info(name)
    log.info(type(row))
    log.info(row.size)
    log.info(col.size)
    log.info(data.size)
    adj_csr = csr_matrix((data, (row, col)), shape=(adj.shape[0], adj.shape[1]))
    
    temp = []
    temp.append(adj_csr.indptr.tolist())
    temp.append(adj_csr.indices.tolist())
    temp.append(adj_csr.data.tolist())
    
    fo = open(name+'.csv', "w")
    writer = csv.writer(fo)
    for r in temp:
        print('ghq')
        writer.writerow(r)
    fo.close()
    
    return True;