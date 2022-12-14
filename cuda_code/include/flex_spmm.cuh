#ifndef FLEX_SPMM_H
#define FLEX_SPMM_H

#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include "common.h"
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
                float* mat_c);

#endif
