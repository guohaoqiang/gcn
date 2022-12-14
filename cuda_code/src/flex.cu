#include "../include/flex.cuh"

void convert(DataLoader& input){
    mat data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	data.csr2tile();
	//data.print1();
	data.print2();

    flexspgemm(data);
    
}
void flexspgemm(mat& data){
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
    
    // Matrix B
    float* mat_b = (float*)malloc(data.m*data.k*sizeof(float));
    for (size_t i=0; i<data.m; ++i){
        for (size_t j=0; j<data.k; ++j){
            mat_b[i*data.k+j] = 1.0;
        }
    }
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
	//cudaMemcpy(d_mat_b, mat_b, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_mat_c, mat_c, data.m*data.k*sizeof(float), cudaMemcpyHostToDevice);

	// each thread block has 2 warps
	dim3 grid(block_tileStart_idx.size()-1, (data.k+31)/32);
    LOG(INFO) << "Ahead the kernel ...";
    std::cout<<"block_tileStart_idx:"<<std::endl;
    print(block_tileStart_idx);
    std::cout<<"warp_tileRow_idx:"<<std::endl;
    print(warp_tileRow_idx);
	
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
	
    float* h_res_c = (float*)malloc(data.m*data.k*sizeof(float)); 
    // transfer data to host
    LOG(INFO) << "Transfer results back ...";
	cudaMemcpy(h_res_c, d_mat_c, data.m*data.k*sizeof(float), cudaMemcpyDeviceToHost);

    // verify results
    LOG(INFO) << "Verify result accuracy ...";
    float* h_ref_c = (float*)malloc(data.m*data.k*sizeof(float)); 
    std::cout<<h_res_c[0]<<std::endl;
    std::cout<<h_res_c[1]<<std::endl;
    std::cout<<h_res_c[3]<<std::endl;
    std::cout<<h_ref_c[0]<<std::endl;
    std::cout<<h_ref_c[1]<<std::endl;
    std::cout<<h_ref_c[3]<<std::endl;
    for (size_t i=0; i<data.m; ++i){
        for (size_t j=0; j<data.k; ++j){
            if (abs(h_ref_c[i*data.k+j]-h_res_c[i*data.k+j])>=0.0001){
                std::cout<<"ref["<<i<<"]["<<j<<"]"<<h_ref_c[i*data.k+j]<<", "<<"gpuC["<<i<<"]["<<j<<"]"<<h_res_c[i*data.k+j]<<std::endl;
            }
        }
    }
}
