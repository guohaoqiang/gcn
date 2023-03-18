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
void flexspgemm(float* h_res_c, MT& data, const float* mat_b, Perfs& perfRes){
	// allocate device memory
    // index of the first nz entry in each tile, length = #tiles+1
    int* d_tileNnz; 
	cudaMalloc(&d_tileNnz, data.nnzPtr.size()*sizeof(int));
    
    // index of the first tile for each thread block, length = #blocks+1
    int* d_block_tileStart_idx; 
	cudaMalloc(&d_block_tileStart_idx, data.block_tileStart_idx.size()*sizeof(int));
    
    // row index of tiles for each thread block, length = #blocks
    int* d_warp_tileRow_idx; 
	cudaMalloc(&d_warp_tileRow_idx, data.warp_tileRow_idx.size()*sizeof(int));
	
    // column index of tiles, length = #tiles
    int* d_tileColIdx; 
	cudaMalloc(&d_tileColIdx, data.tileLeftColIdx.size()*sizeof(int));
     
    // row&col index of vals in sparse matrix, length = nnz
    char* d_r_c_Offset; 
	cudaMalloc(&d_r_c_Offset, data.rc_Offset.size()*sizeof(char));
    
    // non-zero vals of sparse matrix, length = nnz
    float* d_vals; 
	cudaMalloc(&d_vals, data.newVals.size()*sizeof(int));
    
    /*
    // Matrix B
    float* mat_b = (float*)malloc(data.m*data.k*sizeof(float));
    for (size_t i=0; i<data.m; ++i){
        for (size_t j=0; j<data.k; ++j){
            mat_b[i*data.k+j] = 1.0;
        }
    }
    */
    float* d_mat_b; 
	cudaMalloc(&d_mat_b, data.m*data.k*sizeof(float));
    
    // Matrix C
    float* d_mat_c; 
	cudaMalloc(&d_mat_c, data.m*data.k*sizeof(float));
    cudaMemset(d_mat_c, 0.0, data.m*data.k*sizeof(float));
    cudaDeviceSynchronize(); 
    
    
    // transfer data to device
	cudaMemcpy(d_tileNnz, data.nnzPtr.data(), data.nnzPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_block_tileStart_idx, data.block_tileStart_idx.data(), data.block_tileStart_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_warp_tileRow_idx, data.warp_tileRow_idx.data(), data.warp_tileRow_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tileColIdx, data.tileLeftColIdx.data(), data.tileLeftColIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r_c_Offset, data.rc_Offset.data(), data.rc_Offset.size()*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vals, data.newVals.data(), data.newVals.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_b, mat_b, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_mat_c, mat_c, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);

	// each thread block has 2 warps
	dim3 grid(data.block_tileStart_idx.size()-1, (data.k+31)/32);
    LOG(INFO) << "Ahead the kernel ...";
    //std::cout<<"block_tileStart_idx:"<<std::endl;
    //print(block_tileStart_idx);
    //std::cout<<"warp_tileRow_idx:"<<std::endl;
    //print(warp_tileRow_idx);
	
    // warm up
    for (int i=0; i<5; ++i){
        
        flexspgemm_cuda_reg_pre<<<grid, 64>>>(d_tileNnz, 
                                                d_block_tileStart_idx, 
                                                d_warp_tileRow_idx, 
                                                d_tileColIdx, 
                                                data.tileLeftColIdx.size(), 
                                                d_r_c_Offset, 
                                                d_vals, 
                                                data.k, 
                                                d_mat_b, 
                                                d_mat_c);
        
        cudaMemset(d_mat_c, 0.0, data.m*data.k*sizeof(float));
    }
    // run test
    float spgemm_duration;
    cudaEvent_t spgemm_start, spgemm_stop;
	cudaEventCreate(&spgemm_start);
	cudaEventCreate(&spgemm_stop);
    float elap_t = 0; 
    for (int i=0; i<10; ++i){
        cudaEventRecord(spgemm_start);
        /*
        flexspgemm_cuda_reg_pre<<< grid, 64 >>>(d_tileNnz,
                                        d_block_tileStart_idx,
                                        d_warp_tileRow_idx,
                                        d_tileColIdx,
                                        data.tileLeftColIdx.size(),
                                        d_r_c_Offset,
                                        d_vals,
                                        data.k,
                                        d_mat_b,
                                        d_mat_c);
	    */
        cudaEventRecord(spgemm_stop);
	    cudaEventSynchronize(spgemm_stop);
	    cudaEventElapsedTime(&spgemm_duration, spgemm_start, spgemm_stop);
        elap_t += spgemm_duration;
        if (i<9) cudaMemset(d_mat_c, 0.0, data.m*data.k*sizeof(float));
    }
    
    cudaDeviceSynchronize(); 
    LOG(INFO) << "After the kernel ...";

    // transfer data to host
    LOG(INFO) << "Transfer results back ...";
	cudaMemcpy(h_res_c, d_mat_c, data.m*data.k*sizeof(float), cudaMemcpyDeviceToHost);
    
    float t = elap_t*(1e-3)/10;
    perfRes.flex_spgemm_time.push_back(t);
    //std::cout<<"Flexspgemm time: "<<t<<" s "<<std::endl;
    float gflops = (2*data.newVals.size()*data.k)/(1e+9);
    perfRes.flex_spgemm_throughput.push_back(gflops/t);
    //std::cout<<"Flexspgemm Throughput: "<<gflops/t<<" gflops/s "<<std::endl;
    float gb = (float)((data.nnzPtr.size()+data.block_tileStart_idx.size()
            +data.warp_tileRow_idx.size()+data.tileColIdx.size()
            +data.tileLeftColIdx.size()+data.newVals.size()+2*data.m*data.k)*4+data.rc_Offset.size())/(1e+9);
    perfRes.flex_spgemm_bandwidth.push_back(gb/t);
    //std::cout<<"Flexspgemm Bandwidth: "<<gb/t<<" GB/s "<<std::endl;
    
    CHECK_CUDA(cudaFree(d_tileNnz));
	CHECK_CUDA(cudaFree(d_block_tileStart_idx));
	CHECK_CUDA(cudaFree(d_warp_tileRow_idx));
	CHECK_CUDA(cudaFree(d_tileColIdx));
	CHECK_CUDA(cudaFree(d_r_c_Offset));
    CHECK_CUDA(cudaFree(d_vals));
	CHECK_CUDA(cudaFree(d_mat_b));
}


template<typename TP>
void print(vector<TP>& arr){
    for (size_t i=0; i<arr.size(); ++i)
        std::cout << arr[i] << " ";
    std::cout<<std::endl;
}
#endif /* FLEX_H */
