# !/bin/bash

d=4
python profiling_gcn.py -g pubmed -k $d
python profiling_gcn.py -g flickr -k $d
python profiling_gcn.py -g reddit -k $d
python profiling_gcn.py -g ppi -k $d
python profiling_gcn.py -g amazon -k $d
python profiling_gcn.py -g yelp -k $d
