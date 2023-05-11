#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
using namespace std;
struct alignas(16) meta{
    int x,y,z,w;
};
__global__
void flex_cuda_v4(int* tileRowPtr, int* metaTile){
	const int WARPSZ = 32;
	const int lane_id = threadIdx.x % WARPSZ;
    const int warp_id = threadIdx.x / WARPSZ;
	//const uint32_t warps = (blockDim.x + WARPSZ - 1)/WARPSZ;

    //if (blockIdx.x==0 && warp_id==0 && lane_id==0){
        //for (int i=0; i<13; ++i)
        //    printf("@140 tileRowPtr[%d] = %d\n", i,tileRowPtr[i]);
    //}
	
    int computeWidth = 1; // # of C entries to be computed by a thread
	int tileRows_perBlk = 4; // # row tiles per block
    
	for (int row_idx=blockIdx.x*tileRows_perBlk ; row_idx<48; row_idx += (gridDim.x*tileRows_perBlk)){ // over C rows
		
        if (blockIdx.x==1 && warp_id==0 && lane_id==0){
	        int4* ptr = (int4*)tileRowPtr;
	        int4 rowPtr = ptr[row_idx];
            printf("@26:   row_idx = %d, x = %d, y = %d, z = %d, w = %d\n", row_idx, rowPtr.x, rowPtr.y, rowPtr.z, rowPtr.w);
        }
        
        int4 rowPtr = reinterpret_cast<int4*>(tileRowPtr)[row_idx];
        //if (blockIdx.x==1 && warp_id==0 && lane_id==0){
        //    printf("@31:    x = %d, y = %d, z = %d, w = %d\n", rowPtr.x, rowPtr.y, rowPtr.z, rowPtr.w);
        //}
            
        // meta.x: nnzPtr
        // meta.y: #nnz in the current tile (nnzPtr+#nnz == the start of next tile)
        //         MSB bit "1" indicates its the last tile in current row-tiles
        // meta.z: bit_map to mark B rows required by the current tile
        // meta.w: column idx of the current tile. 
        int4 meta = reinterpret_cast<int4*>(metaTile)[rowPtr.x];
        meta = reinterpret_cast<int4*>(metaTile)[rowPtr.y];
        meta = reinterpret_cast<int4*>(metaTile)[rowPtr.z];
        meta = reinterpret_cast<int4*>(metaTile)[rowPtr.w]; 
	} // end C row loops
}

int main(int arc, char** argv){
    // length = 13
    std::vector<int> tileRowPtr{0,3,4,7,10,13,15,16,19,20,30,39,48};
    // length = 48
    std::vector<int> metaTile{};
    for (int i=0; i<48; ++i){
        metaTile.push_back(i);
    }

    for (int i=0; i<13; ++i)
        printf("@62 tileRowPtr[%d] = %d \n", i,tileRowPtr[i]);

    int *d_tileRowPtr;
    cudaMalloc((void **)&d_tileRowPtr, 13*sizeof(int));
    int *d_metaTile;
    cudaMalloc((void **)&d_metaTile, 48*sizeof(int));

    cudaMemcpy(d_tileRowPtr, tileRowPtr.data(), 13*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_metaTile, metaTile.data(), 48*sizeof(int), cudaMemcpyHostToDevice);

    for (int i=0; i<3; ++i){
        flex_cuda_v4<<<12,64>>>(d_metaTile, d_tileRowPtr);
    }
    cudaFree(d_tileRowPtr);
    cudaFree(d_metaTile);
    return EXIT_SUCCESS;
}
