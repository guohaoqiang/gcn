#include "../include/flex.cuh"
/*
void run_test(float* h_res_c, DataLoader& input, 
                const float* mat_b, 
                int tilem,
                int tilen  
                int tilek, 
                int warmup, 
                int runs, 
                Perfs& perfRes){
    
    mat<tilem,tilek> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
    
    //std::cout<<data.vals[0]<<","<<data.vals[32]<<","<<data.vals[64]<<std::endl;
	data.csr2tile();
    //std::cout<<data.newVals[0]<<","<<data.newVals[32]<<","<<data.newVals[64]<<std::endl;
	//data.print1();
	//data.print2();
     
    flexspgemm(h_res_c, data, host_mat_b, perfRes);

}
*/
void run(DataLoader& input){
    Perfs perfRes;
    
    // ------------ run baseline cuspgemm ----------------
    float* host_mat_b = (float*)malloc(input.n*input.dim*sizeof(float)); 
    for (int i=0; i<input.n*input.dim; ++i){
        host_mat_b[i] = input.cpuX[i];
    }
    cuSpgemm(input, perfRes);
    float* h_ref_c = (float*)malloc(input.n*input.dim*sizeof(float)); 
    CUDA_CHECK(cudaMemcpy(h_ref_c, input.gpuRef1, sizeof(float)*input.n*input.dim, cudaMemcpyDeviceToHost));
    // ---------------------------------------------------
/*    
    cudaEventRecord(cuspgemm_stop);
	cudaEventSynchronize(cuspgemm_stop);
	cudaEventElapsedTime(&cuspgemm_duration, cuspgemm_start, cuspgemm_stop);
    float t = cuspgemm_duration*(1e-3)/10;
    std::cout<<"cuSpgemm time: "<<t<<" s "<<std::endl;
    float gflops = (2*input.cpuA->nnz*input.dim)/(1e+9);
    std::cout<<"cuSpgemm Throughput: "<<gflops/t<<" gflops/s "<<std::endl;
    float gb = (float)((input.n+1 + 2*input.cpuA->nnz + 2*input.n*input.dim)*4)/(1e+9);
    std::cout<<"cuSpgemm Bandwidth: "<<gb/t<<" GB/s "<<std::endl;
*/

    // --------- run flex titling spgemm ---------------
    //vector<vector<int>> mnk = {{16,16,16},{32,8,16},{8,32,16}};

    float* h_res_c = (float*)malloc(input.n*input.dim*sizeof(float)); 
    int warmup = 5;
    int runs = 10;
#ifdef CUBE4X4
    {
        mat<4,4> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef CUBE8X8
    {
        mat<8,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef CUBE16X16
    {
        mat<16,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef CUBE32X32
    {
        mat<32,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT8X16
    {
        mat<8,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT16X8
    {
        mat<16,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT8X32
    {
        mat<8,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef REC32X8
    {
        mat<32,8> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT16X32
    {
        mat<16,32> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
#ifdef RECT32X16
    {
        mat<32,16> data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	    data.csr2tile();
        flexspgemm(h_res_c, data, host_mat_b, perfRes);
    
        // verify results
        int count = 0;
        LOG(INFO) << "Verify result accuracy ...";
        for (size_t i=0; i<input.n; ++i){
            for (size_t j=0; j<input.dim; ++j){
                if (abs(h_ref_c[i*input.dim+j]-h_res_c[i*input.dim+j])>=0.001){
                    count++;
                    //if (i<8 && j==0) 
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*input.dim+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*input.dim+j]<<std::endl;
                }
            }
        }
        std::cout<<"Wrong results: "<<count<<std::endl;
        memset(h_res_c, 0, input.n*input.dim*sizeof(float));
    }
#endif
    free(h_res_c);
    free(h_ref_c);
    free(host_mat_b);


    std::cout<<setw(20)<<left<<"sparse mat (M X N) "
        <<setw(18)<<left<<" tile size (tm X tn) "
        <<setw(20)<<left<<"  dense mat (N X K) "
        <<setw(15)<<left<<" cuspgemm t "
        <<setw(15)<<left<<" flex_spgemm t "<<std::endl;
#ifdef CUBE4X4
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 4 X 4 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[0])<<std::endl;
#endif
#ifdef CUBE8X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 8 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[0])<<std::endl;
#endif
#ifdef CUBE16X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 16 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[1])<<std::endl;
#endif
#ifdef CUBE32X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 32 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[2])<<std::endl;
#endif
#ifdef RECT8X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 8 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[0])<<std::endl;
#endif
#ifdef RECT16X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 16 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[0])<<std::endl;
#endif
#ifdef RECT8X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 8 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[0])<<std::endl;
#endif
#ifdef RECT32X8
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 32 X 8 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[0])<<std::endl;
#endif
#ifdef RECT16X32
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 16 X 32 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[0])<<std::endl;
#endif
#ifdef RECT32X16
    std::cout<<setw(20)<<left<<to_string(input.n)+" X "+to_string(input.n)
        <<setw(23)<<left<<" 32 X 16 "
        <<setw(19)<<left<<to_string(input.n)+" X "+to_string(input.dim)
        <<setw(15)<<left<<to_string(perfRes.cuspgemm_time)
        <<setw(15)<<left<<to_string(perfRes.flex_spgemm_time[0])<<std::endl;
#endif
}

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
/*
void flexspgemm(float* h_res_c, const mat& data, const float* mat_b, Perfs& perfRes){

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
    
    
    // Matrix B
    //float* mat_b = (float*)malloc(data.m*data.k*sizeof(float));
    //for (size_t i=0; i<data.m; ++i){
    //    for (size_t j=0; j<data.k; ++j){
    //        mat_b[i*data.k+j] = 1.0;
    //    }
    //}
    
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
    for (size_t i=0; i<5; ++i){
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
    for (size_t i=0; i<10; ++i){
        cudaEventRecord(spgemm_start);
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
	CHECK_CUDA(cudaFree(d_mat_c)); 
}
*/

void cuSpgemm(DataLoader& input, Perfs& perfRes){
    const float alpha = 1.0;
    const float beta = 0.0;

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, input.cpuA->r, input.cpuA->c, input.cpuA->nnz,
                                      input.gpuA->row, input.gpuA->col, input.gpuA->vals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, input.n, input.dim, input.dim, input.gpuX,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, input.n, input.dim, input.dim, input.gpuRef1,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // warm-up
    for (int i=0; i<5; ++i){
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, dBuffer))
    }
    // execute SpMM
    float cuspgemm_duration;
    cudaEvent_t cuspgemm_start, cuspgemm_stop;
	cudaEventCreate(&cuspgemm_start);
	cudaEventCreate(&cuspgemm_stop); 
    float elap_t = 0.0;
    for (int i=0; i<10; ++i){
        cudaEventRecord(cuspgemm_start);
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, dBuffer))
        cudaEventRecord(cuspgemm_stop);
	    cudaEventSynchronize(cuspgemm_stop);
	    cudaEventElapsedTime(&cuspgemm_duration, cuspgemm_start, cuspgemm_stop);
        elap_t += cuspgemm_duration;
    }
    float t = elap_t*(1e-3)/10;
    perfRes.cuspgemm_time = t;
    
    float gflops = (2*input.cpuA->nnz*input.dim)/(1e+9);
    perfRes.cuspgemm_throughput = gflops/t;
    //std::cout<<"cuSpgemm Throughput: "<<gflops/t<<" gflops/s "<<std::endl;
    float gb = (float)((input.n+1 + 2*input.cpuA->nnz + 2*input.n*input.dim)*4)/(1e+9);
    perfRes.cuspgemm_bandwidth = gb/t;
    //std::cout<<"cuSpgemm Bandwidth: "<<gb/t<<" GB/s "<<std::endl;
    
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
}
