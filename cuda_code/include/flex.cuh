#ifndef FLEX_H
#define FLEX_H 
#include <iostream>
#include "common.h"
#include "DataLoader.cuh"
#include "mat.h"
#include "flex_spmm.cuh"
#define CUBE
//#define RECT1
//#define RECT2
/*
void run_test(float* h_res_c, 
                DataLoader& input, 
                const float* mat_b, 
                const vector<vector<int>>& mnk, 
                int idx, 
                int warmup, 
                int runs, 
                Perfs& perfRes);
*/
void run(DataLoader& input);
void cuSpgemm(DataLoader& input, Perfs& perfRes);

template<typename MT>
void flexspgemm(float* h_res_c, MT& data, const float* mat_b, Perfs& perfRes);


template<typename TP>
void print(vector<TP>& arr){
    for (size_t i=0; i<arr.size(); ++i)
        std::cout << arr[i] << " ";
    std::cout<<std::endl;
}
#endif /* FLEX_H */
