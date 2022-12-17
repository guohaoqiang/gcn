#include "../include/flex.cuh"

void convert(DataLoader& input){
    mat data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	data.csr2tile();
	//data.print1();
	data.print2();
    
    float* host_mat_b = (float*)malloc(data.m*data.k*sizeof(float)); 
    for (size_t i=0; i<data.m*data.k; ++i){
        host_mat_b[i] = input.cpuX[i];
    }

    float* h_res_c = (float*)malloc(data.m*data.k*sizeof(float)); 
    flexspgemm(h_res_c, data, host_mat_b);
    cudaDeviceSynchronize();
    
    
    cuSpgemm(input);
    float* h_ref_c = (float*)malloc(data.m*data.k*sizeof(float)); 
    CUDA_CHECK(cudaMemcpy(h_ref_c, input.gpuRef1, sizeof(float)*data.m*data.k, cudaMemcpyDeviceToHost));
    // verify results
    int count = 0;
    LOG(INFO) << "Verify result accuracy ...";
    for (size_t i=0; i<data.m; ++i){
        for (size_t j=0; j<data.k; ++j){
            if (abs(h_ref_c[i*data.k+j]-h_res_c[i*data.k+j])>=0.0001){
                count++;
                if (i<128 && j==0)
                    std::cout<<"ref["<<i<<"]["<<j<<"]="<<h_ref_c[i*data.k+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]="<<h_res_c[i*data.k+j]<<std::endl;
            }
        }
    }
    std::cout<<"Wrong results: "<<count<<std::endl;
}
void flexspgemm(float* h_res_c, mat& data, float* mat_b){
	//uint32_t row_tiles = (data.m+tm-1)/tm;
	uint32_t row_tiles = (data.m+4-1)/4;

	// workload_per_block: tiles assigned to a block
	uint32_t workload_per_block = 4;
	vector<uint32_t> block_tileStart_idx;
	vector<uint32_t> warp_tileRow_idx;
	// block_tileStart_idx[i]: the idx of the begining tile assigned to the i-th block
	// the last one is the totall number of tiles
	for (size_t i=0; i<row_tiles; ++i){
		uint32_t blocks_req = (data.tileRowPtr[i+1] - data.tileRowPtr[i] + workload_per_block-1)/workload_per_block;
		for (size_t k=0; k<blocks_req; ++k){
			block_tileStart_idx.push_back(data.tileRowPtr[i] + k*workload_per_block);
			// tile row idx for tiles assigned to each thread block
			warp_tileRow_idx.push_back(i*TM);
		}
	}
	block_tileStart_idx.push_back(data.tileRowPtr.back());


	// allocate device memory
    // index of the first nz entry in each tile, length = #tiles+1
    int* d_tileNnz; 
	cudaMalloc(&d_tileNnz, data.nnzPtr.size()*sizeof(int));
    
    // index of the first tile for each thread block, length = #blocks+1
    int* d_block_tileStart_idx; 
	cudaMalloc(&d_block_tileStart_idx, block_tileStart_idx.size()*sizeof(int));
    
    // row index of tiles for each thread block, length = #blocks
    int* d_warp_tileRow_idx; 
	cudaMalloc(&d_warp_tileRow_idx, warp_tileRow_idx.size()*sizeof(int));
	
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
    
    
    
    // transfer data to device
	cudaMemcpy(d_tileNnz, data.nnzPtr.data(), data.nnzPtr.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_block_tileStart_idx, block_tileStart_idx.data(), block_tileStart_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_warp_tileRow_idx, warp_tileRow_idx.data(), warp_tileRow_idx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tileColIdx, data.tileLeftColIdx.data(), data.tileLeftColIdx.size()*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r_c_Offset, data.rc_Offset.data(), data.rc_Offset.size()*sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_vals, data.newVals.data(), data.newVals.size()*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mat_b, mat_b, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_mat_c, mat_c, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);

	// each thread block has 2 warps
	dim3 grid(block_tileStart_idx.size()-1, (data.k+31)/32);
    LOG(INFO) << "Ahead the kernel ...";
    //std::cout<<"block_tileStart_idx:"<<std::endl;
    //print(block_tileStart_idx);
    //std::cout<<"warp_tileRow_idx:"<<std::endl;
    //print(warp_tileRow_idx);
	
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
    LOG(INFO) << "After the kernel ...";

    // transfer data to host
    LOG(INFO) << "Transfer results back ...";
	cudaMemcpy(h_res_c, d_mat_c, data.m*data.k*sizeof(float), cudaMemcpyDeviceToHost);
}


void cuSpgemm(DataLoader& input){
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

    // execute SpMM
    CHECK_CUSPARSE(cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG3, dBuffer))
}
