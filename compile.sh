#!/bin/bash

nvcc -std c++20 -Xcompiler -fPIC -shared renumber.cu -o renumber.so
nvcc -std c++20 -Xcompiler -fPIC -shared tile.cu -o tile.so
nvcc -Xcompiler -fPIC -shared -o permutate.so permutate.cu
nvcc -Xcompiler -fPIC -shared -o flexspmm.so flexspmm.cu

#nvcc -Xcompiler -fPIC -shared -o cuspmm.so cuspmm.cu -lcusparse
