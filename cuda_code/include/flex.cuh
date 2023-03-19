#ifndef FLEX_H
#define FLEX_H 
#include <iostream>
#include "common.h"
#include "DataLoader.cuh"
#include "mat.h"
#include "flex_spmm.cuh"
//#define CUBE4X4
//#define CUBE8X8
#define CUBE16X16
//#define CUBE32X32
//#define RECT8X16
//#define RECT16X8
//#define RECT8X32
//#define RECT32X8
//#define RECT16X32
//#define RECT32X16
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
