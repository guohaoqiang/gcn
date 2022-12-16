#ifndef FLEX_H
#define FLEX_H 
#include <iostream>
#include "common.h"
#include "DataLoader.cuh"
#include "mat.h"
#include "flex_spmm.cuh"

void convert(DataLoader& input);
void cuSpgemm(DataLoader& input);


void flexspgemm(float* res, mat& data, float* mat_b);


template<typename TP>
void print(vector<TP>& arr){
    for (size_t i=0; i<arr.size(); ++i)
        std::cout << arr[i] << " ";
    std::cout<<std::endl;
}
#endif /* FLEX_H */
