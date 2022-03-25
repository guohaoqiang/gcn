import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil, floor, sqrt
import glog as log

def load(f):
    for i in f:
        return i

def get_nnz(f):
    row_offset = load(f)
    ans = []
    for i in range(1,len(row_offset)):
        ans.append(int(row_offset[i])-int(row_offset[i-1]))
    log.info(len(ans))
    return ans;
def get_xy(r):
    a = pd.Series(r)
    b = a.value_counts()
    yy = [] # Frequency
    for i in list(b.index):
        yy.append(b[i])
    yy = np.array(yy)
    yy = yy/yy.sum()  # Percentile
    xx = b.index      # NNZs
    ind = np.lexsort((yy,xx))
    x = [0]
    y = [0]
    
    for i in ind:
        x.append(xx[i])
        y.append(yy[i]+y[-1])
    return min(x),max(x),x,y

f = open('cora.csv','r')
f1 = csv.reader(f)
cora = get_nnz(f1)
f.close()

f = open('polblogs.csv','r')
f2 = csv.reader(f)
polblogs = get_nnz(f2)
f.close()

f = open('citeseer.csv','r')
f3 = csv.reader(f)
citeseer = get_nnz(f3)
f.close()

f = open('pubmed.csv','r')
f4 = csv.reader(f)
pubmed = get_nnz(f4)
f.close()

f = open('ppi.csv','r')
f5 = csv.reader(f)
ppi = get_nnz(f5)
f.close()

f = open('flickr.csv','r')
f6 = csv.reader(f)
flickr = get_nnz(f6)
f.close()

f = open('reddit.csv','r')
f7 = csv.reader(f)
reddit = get_nnz(f7)
f.close()

f = open('yelp.csv','r')
f8 = csv.reader(f)
yelp = get_nnz(f8)
f.close()

f = open('amazon.csv','r')
f9 = csv.reader(f)
amazon = get_nnz(f9)
f.close()

fig = plt.gcf()
fig.set_size_inches(12,11)


cora_mn, cora_mx, cora_x, cora_y = get_xy(cora)
polblogs_mn, polblogs_mx, polblogs_x, polblogs_y = get_xy(polblogs)
citeseer_mn, citeseer_mx, citeseer_x, citeseer_y = get_xy(citeseer)
pubmed_mn, pubmed_mx, pubmed_x, pubmed_y = get_xy(pubmed)
ppi_mn, ppi_mx, ppi_x, ppi_y = get_xy(ppi)
flickr_mn, flickr_mx, flickr_x, flickr_y = get_xy(flickr)
reddit_mn, reddit_mx, reddit_x, reddit_y = get_xy(reddit)
yelp_mn, yelp_mx, yelp_x, yelp_y = get_xy(yelp)
amazon_mn, amazon_mx, amazon_x, amazon_y = get_xy(amazon)
mn = min([cora_mn,polblogs_mn,citeseer_mn,pubmed_mn,ppi_mn,flickr_mn,reddit_mn,yelp_mn,amazon_mn])
mx = max([cora_mx,polblogs_mx,citeseer_mx,pubmed_mx,ppi_mx,flickr_mx,reddit_mx,yelp_mx,amazon_mx])


plt.step(cora_y, np.log10(cora_x), label='cora')
log.info(cora_mn)
log.info(cora_mx)

plt.step(polblogs_y, np.log10(polblogs_x), label='polblogs')
log.info(polblogs_mn)
log.info(polblogs_mx)

plt.step(citeseer_y, np.log10(citeseer_x), label='citeseer')
log.info(citeseer_mn)
log.info(citeseer_mx)

plt.step(pubmed_y, np.log10(pubmed_x), label='pubmed')
log.info(pubmed_mn)
log.info(pubmed_mx)

plt.step(ppi_y, np.log10(ppi_x), label='ppi')
log.info(ppi_mn)
log.info(ppi_mx)

plt.step(flickr_y, np.log10(flickr_x), label='flickr')
log.info(flickr_mn)
log.info(flickr_mx)

plt.step(reddit_y, np.log10(reddit_x), label='reddit')
log.info(reddit_mn)
log.info(reddit_mx)

plt.step(yelp_y, np.log10(yelp_x), label='yelp')
log.info(yelp_mn)
log.info(yelp_mx)

plt.step(amazon_y, np.log10(amazon_x), label='amazon')
log.info(amazon_mn)
log.info(amazon_mx)

plt.title('Cumulative distribution function of NNZs in each row')

plt.ylabel('NNZs in each row (log10)')

plt.xlabel('Percentile of NNZ-row')

plt.legend(loc='best')

plt.xlim([-0.01,1.01])

plt.ylim([np.log10(1), np.log10(mx)+0.1])

plt.grid()

plt.savefig("row.svg")