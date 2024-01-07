#include <cuda_runtime.h>
extern "C"{
__global__
void flexspmm_v9_permuteX(float* mat_b_dev, float* shadow_b_dev, int* voMp_dev,
                            int* labels_dev, int* shadow_labels_dev, 
                            int m, int n, int k){
    // preprocess dense mat B. out-of-place permutation of B rows
    const int rows_p_blk = blockDim.x / 32; // a warp moves a row
    const int lane_id = threadIdx.x % 32;
	for (int row_idx=blockIdx.x*rows_p_blk+threadIdx.x/32; 
            row_idx<n; row_idx += (gridDim.x*rows_p_blk)){ // over C rows
      
        int tgt_row = voMp_dev[row_idx];  
        for (int i = lane_id; i<k; i += 32){
            shadow_b_dev[ row_idx*k+i ] = mat_b_dev[ tgt_row*k+i ]; 
        }
        if (lane_id==0){
            shadow_labels_dev[ row_idx] = labels_dev[ tgt_row ]; 
        }
	} // end C row loops    
}
__global__
void put_back(float* mat_b_dev, float* shadow_b_dev,
              int* labels_dev, int* shadow_labels_dev, 
              int m, int n, int k){
    // preprocess dense mat B. out-of-place permutation of B rows
    const int rows_p_blk = blockDim.x / 32; // a warp moves a row
    const int lane_id = threadIdx.x % 32;
	for (int row_idx=blockIdx.x*rows_p_blk+threadIdx.x/32; 
            row_idx<n; row_idx += (gridDim.x*rows_p_blk)){ // over C rows
      
        for (int i = lane_id; i<k; i += 32){
            mat_b_dev[ row_idx*k+i ] = shadow_b_dev[ row_idx*k+i ]; 
        }
        if (lane_id==0){
            labels_dev[ row_idx ] = shadow_labels_dev[ row_idx ]; 
        }
	} // end C row loops    
}
void permutate(float* mat_b_dev, int* voMp_dev, int* labels_dev, 
                int m, int n, int k){
    
    float* shadow_b_dev; 
    cudaMalloc( &shadow_b_dev, sizeof(float)*n*k );
    cudaMemset( shadow_b_dev, 0, sizeof(float)*n*k );
    int* shadow_labels_dev; 
    cudaMalloc( &shadow_labels_dev, sizeof(int)*m );
    cudaMemset( shadow_labels_dev, 0, sizeof(int)*m );
    flexspmm_v9_permuteX<<<1024, 128>>>(mat_b_dev, shadow_b_dev, voMp_dev,
                                        labels_dev, shadow_labels_dev, 
                                        m, n, k);
    cudaDeviceSynchronize();
    put_back<<<1024, 128>>>(mat_b_dev, shadow_b_dev,
                            labels_dev, shadow_labels_dev, 
                            m, n, k);
    cudaDeviceSynchronize();
    cudaFree(shadow_b_dev);
    cudaFree(shadow_labels_dev);
}
}
