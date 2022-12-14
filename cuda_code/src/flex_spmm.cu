#include "../include/flex_spmm.cuh"
__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );
    return addr;
}
__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}
__device__ __forceinline__
void sts8(const uint8_t &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.u8 [%0], %1;\n"
        : : "r"(addr), "r"((int)reg)
    );
}
__device__ __forceinline__
void lds8(uint8_t &reg, const uint32_t &addr) {
    int temp = reg;
    asm volatile (
        "ld.shared.u8 %0, [%1];\n"
        : "=r"(temp)
        : "r"(addr)
    );
    reg = temp;
}
__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}
__device__ __forceinline__
void lds32(float &reg, const uint32_t &addr) {
    asm volatile (
        "ld.shared.f32 %0, [%1];\n"
        : "=f"(reg)
        : "r"(addr)
    );
}

// inputs:
//		 block_tileStart_idx: the tile idex of the first tile computed by thread-blocks
//       tileColIdx: the column idex of the first column of each tile
//       tileNnz: the index of the first nze(non-zero entry) in each tile
//       warp_tileRow_idx: row idx of each tile
//       tiles: # of tiles
//       r_c_Offset(8 bits): the row index (upper 4 bits) + column index (lower 4 bits) of each nze in each tile
//       vals: non zero entries
// The real row-col index for a nze (the i-th row tile, the j-th tile of the i-th row tile): 
//  r = i + r_c_Offset[7...4], c = tileColIdx[ tileRowPtr[i] + j ] + r_c_Offset[3...0]

// A: sparse, m * n
// B: dense, n * k   (k << n)
__global__
void flexspgemm_cuda_reg_pre(int* tileNnz,
				int* block_tileStart_idx,
				int* warp_tileRow_idx,
                int* tileColIdx,
				int tiles,
				char* r_c_Offset,
				float* vals,
				int k,
				float* mat_b,
                float* mat_c){
	const uint32_t WARPSZ = 32;
	const uint32_t lane_id = threadIdx.x % WARPSZ;
	const uint32_t warp_id = threadIdx.x / WARPSZ;
	const uint32_t warps = (blockDim.x + WARPSZ - 1)/WARPSZ;
	
	int tile_ref[3] = {-1};
	if (blockIdx.x | 0){
		tile_ref[0] = block_tileStart_idx[blockIdx.x-1];
	}
	tile_ref[1] = block_tileStart_idx[blockIdx.x];
	tile_ref[2] = block_tileStart_idx[blockIdx.x+1];
	const uint32_t warp_tileStart_id = tile_ref[1] + warp_id;
	
	// matrix A: 
	//          non-zero vals: tm*tn * 4bytes * double buffer <= 16*16 * 4 * 2 = 2k
	//          non-zero entry idx: tm*tn * 1bytes * double buffer <= 16*16 * 1 * 2 = 0.5k
	// matrix B:
	//          each thread has 16 regs to store one B column segment
	// Only A is stored in shared memory
	/*  
		| ----- vals ----- | -- idx -- | ----- vals ----- | -- idx -- |
		|       1k         |   256B    |       1k         |   256B    |     
		a_vals_smem ^= 0x0500
		a_idx_smem  ^= 0x0500
	*/
	// 2 warps in a block
	#define WARPS 2
	#define ACC_SH  true
	#if ACC_SH
	__shared__ char smem[(8*16*16 + 512)*WARPS+16*32*4];
	#else
	__shared__ char smem[(8*16*16 + 512)*WARPS];
	#endif
	float* a_vals_smem = reinterpret_cast<float *>(smem+warp_id*2560);                 // 2.5k
	uint8_t* a_idx_smem = reinterpret_cast<uint8_t *>(smem+warp_id*2560 + 1024);
	float* c_mat_sh = reinterpret_cast<float *>(smem+WARPS*2560);                     // two warps: 2 * 2.5k = 5k

	uint32_t a_vals_sts = smem_u32addr(a_vals_smem);
	uint32_t a_vals_lds = smem_u32addr(a_vals_smem);
	uint32_t a_idx_sts = smem_u32addr(a_idx_smem);
	uint32_t a_idx_lds = smem_u32addr(a_idx_smem);

	// ************ load A tile ( && B rows required by first tile) from glb mem to shared memory (registers) *********************************************
	// row_flag[0]: bits represent existance of B rows, determined by col idx of A entrys 
	// row_flag[1]: bits represent existance of C rows, determined by row idx of A entrys 
	uint32_t row_flag[2] = {0};
	
  	float b_reg[2][16];
	// both tileNNz and r_c_offfset have good locality, so no need to optimize their memory access behavior? 
	#define FULL_MASK 0xffffffff
	uint32_t steps = 1;
    //printf("kernel begins ...\n");	
	for (uint32_t entry_idx = tileNnz[warp_tileStart_id]; entry_idx<tileNnz[warp_tileStart_id+1]; entry_idx += steps){
		
		if (tileNnz[warp_tileStart_id+1]-entry_idx>=32){         // if more than 32 non-zero entrys left in current tile
			int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
			uint32_t r_offset = rc_idx & 240;     // .. & 1111 0000
			uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111
			// load a to shared mem
			uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
			sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
            float aVal_tmp = vals[entry_idx+lane_id];
			sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

			// load b to registers
			for (uint32_t j=0; j<32; ++j){
				int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j);
				//char rc_idx_tmp = a_idx_sts[entry_idx+j-tileNnz[warp_tileStart_id]];

				//r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
				c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
				// the i-th bit 1 represents the i-th B row is alrealdy loaded in shared memory
				if ((row_flag[0] & (1<<c_offset)) == 0 ){
					// mark it as loaded
					row_flag[0] |= (1<<c_offset);

					uint32_t entry_col_idx = tileColIdx[warp_tileStart_id] + c_offset;  // .. & 0000 1111
					uint32_t lane_offset = blockIdx.y*32 + lane_id;
					if (lane_offset<min(blockIdx.y*32+32, k)){
						// matrix B is in row major
						// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
						b_reg[warp_tileStart_id%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
					}
				}
			}
			steps = 32;
		}else if (tileNnz[warp_tileStart_id+1]-entry_idx>=16){
			int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
			uint32_t r_offset = rc_idx & 240;     // .. & 1111 0000
			uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111
			//uint32_t mask = __ballot_sync(FULL_MASK, entry_idx<tileNnz[warp_tileStart_id+1]);
			//uint32_t act_thds = __popc(mask);

			// load a to shared mem
			uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
			sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
            float aVal_tmp = vals[entry_idx+lane_id];
			sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));
			

			// load b to registers
			for (uint32_t j=0; j<16; ++j){
				int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
				//char rc_idx_tmp = a_idx_sts[entry_idx+j-tileNnz[warp_tileStart_id]];
				//r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
				c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
				// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
				if ((row_flag[0] & (1<<c_offset)) == 0 ){
					// mark it as loaded
					row_flag[0] |= (1<<c_offset);

					uint32_t entry_col_idx = tileColIdx[warp_tileStart_id] + c_offset;  // .. & 0000 1111
					uint32_t lane_offset = blockIdx.y*32 + lane_id;
					if (lane_offset<min(blockIdx.y*32+32, k)){
						// matrix B is in row major
						// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
						b_reg[warp_tileStart_id%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
					}
				}
			}
			steps = 16;
		}else if (tileNnz[warp_tileStart_id+1]-entry_idx>=8){
			int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
			uint32_t r_offset = rc_idx & 240;     // .. & 1111 0000
			uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111
			//uint32_t mask = __ballot_sync(FULL_MASK, entry_idx<tileNnz[warp_tileStart_id+1]);
			//uint32_t act_thds = __popc(mask);

			// load a to shared mem
			uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
			sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
            float aVal_tmp = vals[entry_idx+lane_id];
			sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));
			
            // load b to registers
			for (uint32_t j=0; j<8; ++j){
				int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
				//char rc_idx_tmp = a_idx_sts[entry_idx+j-tileNnz[warp_tileStart_id]];
				//r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
				c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
				// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
				if ((row_flag[0] & (1<<c_offset)) == 0 ){
					// mark it as loaded
					row_flag[0] |= (1<<c_offset);

					uint32_t entry_col_idx = tileColIdx[warp_tileStart_id] + c_offset;  // .. & 0000 1111
					uint32_t lane_offset = blockIdx.y*32 + lane_id;
					if (lane_offset<min(blockIdx.y*32+32, k)){
						// matrix B is in row major
						// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
						b_reg[warp_tileStart_id%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
					}
				}
			}
			steps = 8;
		}else{
			int rc_idx = r_c_Offset[entry_idx];   // in coalesced way
			uint32_t r_offset = rc_idx & 240;     // .. & 1111 0000
			uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111

			// load a to shared mem
			uint8_t aVal_idx_tmp = r_c_Offset[entry_idx];
			sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
            float aVal_tmp = vals[entry_idx];
			sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

			// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
			// load b to registers
			if ((row_flag[0] & (1<<c_offset)) == 0 ){
				// mark it as loaded
				row_flag[0] |= (1<<c_offset);

				uint32_t entry_col_idx = tileColIdx[warp_tileStart_id] + c_offset;  // .. & 0000 1111
				uint32_t lane_offset = blockIdx.y*32 + lane_id;
				if (lane_offset<min(blockIdx.y*32+32, k)){
					// matrix B is in row major
					// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
					b_reg[warp_tileStart_id%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
				}
			}
			steps = 1;
		}
	}
	row_flag[0] = 0;
	a_vals_sts ^= 0x0500;
	a_idx_sts  ^= 0x0500;
	// ***************************************************************************************************************************************************

	// multiplication loops
	
	float res[16] = {0};
	
	// iterate tiles assigned to the current block
	for (uint32_t i=warp_tileStart_id; i<tile_ref[2]; i += warps){
	    int  nnz_cur_tile = tileNnz[i+1]-tileNnz[i];
        if (blockIdx.x == 0){
            //printf("Using CUDA cores\n");
            //printf("warps = %d\n",warps);
            //printf("tileref[0] = %d,tileref[0] = %d,tileref[0] = %d\n",tile_ref[0],tile_ref[1],tile_ref[2]);
            printf("tileID = %d, nnz = %d\n",i,nnz_cur_tile);
        }
		
		// ************ load B rows required by "next tile" from glb mem to shmem **********
		// both tileNNz and r_c_offfset have good locality, so no need to optimize their memory access behavior?
		if (i+warps<tile_ref[2]){
			//uint32_t boundry = tileNnz[i+warps+1]; 
			for (uint32_t entry_idx = tileNnz[i+warps]; entry_idx<tileNnz[i+warps+1]; entry_idx += steps){
				
				if (tileNnz[i+warps+1]-entry_idx>=32){
					int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
					uint32_t r_offset = rc_idx & 240;     // .. & 1111 0000
					uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111

					// load a to shared mem
			        uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
			        sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
                    float aVal_tmp = vals[entry_idx+lane_id];
			        sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

					// load b to registers
					for (uint32_t j=0; j<32; ++j){
						int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
						//r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
						c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
						// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
						if ((row_flag[0] & (1<<c_offset)) == 0 ){
							// mark it as loaded
							row_flag[0] |= (1<<c_offset);

							uint32_t entry_col_idx = tileColIdx[i+warps] + c_offset;  // .. & 0000 1111
							uint32_t lane_offset = blockIdx.y*32 + lane_id;
							if (lane_offset<min(blockIdx.y*32+32, k)){
								// matrix B is in row major
								// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
								b_reg[(i+1)%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
							}
						}
					}
					steps = 32;
				}else if (tileNnz[i+warps+1]-entry_idx>=16){
					int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
					uint32_t r_offset = rc_idx & 240;     // .. & 1111 0000
					uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111

					// load a to shared mem
			        uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
			        sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
                    float aVal_tmp = vals[entry_idx+lane_id];
			        sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

					// load b to registers
					for (uint32_t j=0; j<16; ++j){
						int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
						// r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
						c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
						// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
						if ((row_flag[0] & (1<<c_offset)) == 0 ){
							// mark it as loaded
							row_flag[0] |= (1<<c_offset);

							uint32_t entry_col_idx = tileColIdx[i+warps] + c_offset;  // .. & 0000 1111
							uint32_t lane_offset = blockIdx.y*32 + lane_id;
							if (lane_offset<min(blockIdx.y*32+32, k)){
								// matrix B is in row major
								// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
								b_reg[(i+1)%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
							}
						}
					}
					steps = 16;
				}else if (tileNnz[i+warps+1]-entry_idx>=8){
					int rc_idx = r_c_Offset[entry_idx+lane_id]; // in coalesced way
					uint32_t r_offset = rc_idx & 240;     // .. & 1111 0000
					uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111

					// load a to shared mem
			        uint8_t aVal_idx_tmp = r_c_Offset[entry_idx+lane_id];
			        sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
                    float aVal_tmp = vals[entry_idx+lane_id];
			        sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

					// load b to registers
					for (uint32_t j=0; j<8; ++j){
						int rc_idx_tmp = __shfl_sync(FULL_MASK, rc_idx, j); 
						// r_offset = rc_idx_tmp & 240;     // .. & 1111 0000
						c_offset = rc_idx_tmp & 15;      // .. & 0000 1111
						// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
						if ((row_flag[0] & (1<<c_offset)) == 0 ){
							// mark it as loaded
							row_flag[0] |= (1<<c_offset);

							uint32_t entry_col_idx = tileColIdx[i+warps] + c_offset;  // .. & 0000 1111
							uint32_t lane_offset = blockIdx.y*32 + lane_id;
							if (lane_offset<min(blockIdx.y*32+32, k)){
								// matrix B is in row major
								// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
								b_reg[(i+1)%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
							}
						}
					}		
					steps = 8;
				}else{
					int rc_idx = r_c_Offset[entry_idx];   // broadcast
					uint32_t r_offset = rc_idx & 240;     // .. & 1111 0000
					uint32_t c_offset = rc_idx & 15;      // .. & 0000 1111
					// load a to shared mem
			        uint8_t aVal_idx_tmp = r_c_Offset[entry_idx];
			        sts8(aVal_idx_tmp, a_idx_sts + (entry_idx+lane_id-tileNnz[warp_tileStart_id])*sizeof(uint8_t));
                    float aVal_tmp = vals[entry_idx];
			        sts32(aVal_tmp, a_vals_sts + (r_offset*16+c_offset)*sizeof(float));

					// load b to registers
					// the i-th bit represents the i-th B row is alrealdy loaded in shared memory
					if ((row_flag[0] & (1<<c_offset)) == 0 ){
						// mark it as loaded
						row_flag[0] |= (1<<c_offset);

						uint32_t entry_col_idx = tileColIdx[i+warps] + c_offset;  // .. & 0000 1111
						uint32_t lane_offset = blockIdx.y*32 + lane_id;
						if (lane_offset<min(blockIdx.y*32+32, k)){
							// matrix B is in row major
							// b_mat_sh[warp_id*2 + warp_tileStart_id%2][c_offset*tn+lane_id] = mat_b[entry_col_idx*k + lane_offset];
							b_reg[(i+1)%2][c_offset] = mat_b[entry_col_idx*k + lane_offset];
						}
					}
					steps = 1;
				}
			}
			row_flag[0] = 0;
			a_vals_sts ^= 0x0500;
			a_idx_sts  ^= 0x0500;
		} // end if
		// ************************************************************************************

	    float a_reg[2] = {0.0};
		if (nnz_cur_tile < 1*4*4){
            // Cuda cores
			
			// visit all nze in the current tile
			// both tileNNz and r_c_offfset have good locality, so no need to optimize their memory access behavior?
            uint8_t a_idx_tmp;
            lds8(a_idx_tmp, a_idx_lds + 0*sizeof(uint8_t));
			//uint32_t r_offset = a_idx_lds[0] & 240;     // .. & 1111 0000
			//uint32_t c_offset = a_idx_lds[0] & 15;      // .. & 0000 1111
			uint32_t r_offset = a_idx_tmp & 240;     // .. & 1111 0000
			uint32_t c_offset = a_idx_tmp & 15;      // .. & 0000 1111
			//a_reg[tileNnz[i]%2] = a_vals_lds[r_offset*16+c_offset];
			lds32(a_reg[tileNnz[i]%2], a_vals_lds + (r_offset*16+c_offset)*sizeof(float));
			for (uint32_t entry_idx = tileNnz[i]; entry_idx<tileNnz[i+1]; ++entry_idx){
				uint32_t r_offset_tmp, c_offset_tmp;

				// preload A vals
				if ((entry_idx+1)<tileNnz[i+1]){
                    uint8_t a_idx_tmp;
                    lds8(a_idx_tmp, a_idx_lds + (entry_idx+1-tileNnz[i])*sizeof(uint8_t));
					//r_offset_tmp = a_idx_lds[entry_idx+1-tileNnz[i]] & 240;     // .. & 1111 0000
					//c_offset_tmp = a_idx_lds[entry_idx+1-tileNnz[i]] & 15;      // .. & 0000 1111
					r_offset_tmp = a_idx_tmp & 240;     // .. & 1111 0000
					c_offset_tmp = a_idx_tmp & 15;      // .. & 0000 1111
					//a_reg[(entry_idx+1)%2] = a_vals_lds[r_offset_tmp*16+c_offset_tmp];
			        lds32(a_reg[(entry_idx+1)%2], a_vals_lds + (r_offset_tmp*16+c_offset_tmp)*sizeof(float));
				}
				// bits to mark C rows to write back
				// here if condition can be removed
				if ((row_flag[1] & (1<<r_offset)) == 0){
					row_flag[1] |= (1<<r_offset);
				}
				
				
				uint32_t lane_offset = blockIdx.y*32 + lane_id;
				if (lane_offset<min(blockIdx.y*32+32, k)){
					// multiplication
					// accumulate in local registers
					res[r_offset] += a_reg[entry_idx%2] * b_reg[i%2][c_offset];	
				}
				r_offset = r_offset_tmp;
				c_offset = c_offset_tmp;
			}
			a_vals_lds ^= 0x0500;
			a_idx_lds  ^= 0x0500;
		}else{
            printf("Using tensor cores\n");
            printf("tileID = %d, nnz = %d\n",i,nnz_cur_tile);
			// Tensor cores
		}
	}
	
	if (ACC_SH){
		// accumulate partial products on shared memory among warps within one thread block
		// w/o bank conflict since a thread access one bank
		// here row_flags[1] helps to reduce atomicAdd
		uint32_t lane_offset = blockIdx.y*32 + lane_id;
		if (lane_offset<min(blockIdx.y*32+32, k)){
			for (uint32_t j=0; j<16 && (row_flag[1] & (1<<j)); ++j){
				atomicAdd(&c_mat_sh[j*32+lane_id], res[j]);
                //printf("@466 : r = %d, c = %d, val = %f\n",j,lane_id,res[j]);
			}
		}
		__syncthreads();
	}
	
	// no need synchronization? because no corporation among warps
	// __syncthreads();
	if (!ACC_SH){
		uint32_t lane_offset = blockIdx.y*32 + lane_id;
		if (lane_offset<min(blockIdx.y*32+32, k)){
			for (uint32_t i=0; i<16 && (row_flag[1] & (1<<i)); ++i){
				uint32_t r = tile_ref[1] + i;
				atomicAdd(&mat_c[r*k+lane_offset], res[i]);
			}
		}
        //printf("Not acc in shared mem\n");	
	}else{
		// transfer results from shared mem to glb mem
		if((tile_ref[1] && (tile_ref[1]==tile_ref[0])) || tile_ref[1]==tile_ref[2]){
			// multi blocks work on one row tiles
			// global memory atomic write, in coalesced way
			uint32_t lane_offset = blockIdx.y*32 + lane_id;
			if (lane_offset<min(blockIdx.y*32+32, k)){
				// each warp transfer one row segment of C
				for (uint32_t i=warp_id; i<16 && (row_flag[1] & (1<<i)); i+=WARPS){
					uint32_t r = tile_ref[1] + i;
					atomicAdd(&mat_c[r*k+lane_offset], c_mat_sh[i*32+lane_id]);
				}
			}
            //printf("Acc1 in shared mem\n");	
		}else{
			// global memory write, in coalesced way
			// transfer results from shared to glb
			uint32_t lane_offset = blockIdx.y*32 + lane_id;
			if (lane_offset<min(blockIdx.y*32+32, k)){
				// each warp transfer one row segment of C
				for (uint32_t i=warp_id; i<16 && (row_flag[1] & (1<<i)); i+=WARPS){
					uint32_t r = tile_ref[1] + i;
					mat_c[r*k+lane_offset] = c_mat_sh[i*32+lane_id];
                    //printf("r = %d, c = %d, val = %f\n",r,lane_offset,mat_c[r*k+lane_offset]);
				}
                //printf("Acc2 in shared mem\n");	
			}	
		}
	}
	return ;
}
