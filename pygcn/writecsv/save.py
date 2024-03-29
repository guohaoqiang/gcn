import logging as log
import csv
import torch
import numpy as np
from scipy.sparse import csr_matrix

def write(adj,name):
    log.info(name)
    log.info(type(adj))
    log.info(adj.shape)
    
    adj_coo = adj.to("cpu")#.tolist()
    temp = adj_coo.coalesce()
    
    row = np.array(temp.indices().tolist()[0])
    col = np.array(temp.indices().tolist()[1])
    data = np.array(temp.values().tolist())
    #row = np.array(temp.indices().tolist()[0])
    #col = np.array(temp.indices().tolist()[1])
    #data = np.array(temp.values().tolist())
    temp = 0
    
    log.info(name)
    log.info(type(row))
    log.info(row.size)
    log.info(col.size)
    log.info(data.size)
    
    if name=="amazon":
        
        fo = open(name+'3.csv', "a+")
        writer = csv.writer(fo)
        adj_csr = csr_matrix((data, (row, col)), shape=(adj.shape[0], adj.shape[1]))
        data = 0
        row = 0
        col = 0
        log.info("coo->csr finished.")
        #temp = adj_csr.indptr.tolist()
        #log.info(len(temp))
        #writer.writerow(temp)
        #log.info("row offset appended.")
        #temp = adj_csr.indices.tolist()
        #log.info(len(temp))
        #writer.writerow(temp)
        #log.info("col index appended.")
        temp = adj_csr.data.tolist()
        log.info(len(temp))
        writer.writerow(temp)
        log.info("Value appended.")
        fo.close()
        return True
        
        '''
        fo = open(name+'_coo'+'.csv', "w")
        writer = csv.writer(fo)
        writer.writerow(row.tolist())
        writer.writerow(col.tolist())
        writer.writerow(data.tolist())
        fo.close()
        return True
        '''
    adj_csr = csr_matrix((data, (row, col)), shape=(adj.shape[0], adj.shape[1]))
    log.info("coo->csr finished.")
    temp = []
    temp.append(adj_csr.indptr.tolist())
    log.info("row offset appended.")
    temp.append(adj_csr.indices.tolist())
    log.info("col index appended.")
    temp.append(adj_csr.data.tolist())
    log.info("Value appended.")
    
    
    fo = open(name+'.csv', "w")
    writer = csv.writer(fo)
    for r in temp:
        print('ghq')
        writer.writerow(r)
    fo.close()
    
    return True
