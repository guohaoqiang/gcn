#include "../include/flex.h"

void convert(DataLoader& input){
    mat data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.dim, input.cpuA->nnz);
	data.csr2tile();
	//data.print1();
	data.print2();

    flexspgemm(data);
    
}

void flexspgem(mat& data){
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
			warp_tileRow_idx.push_back(i);
		}
	}
	block_tileStart_idx.push_back(data.tileRowPtr.back());


	// allocate device memory
	cudaMalloc();
	// transfer data to device
	cudaMemcpy();

	// each thread block has 2 warps
	dim3 grid(block_tileStart_idx.size()-1, (data.k+31)/32);
	flexspgem_cuda_reg_pre<<<grid, 64>>>();
}
