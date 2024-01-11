#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>
#define tm 8

extern "C"{

// Return Streaming Multiprocessor (aka SM, MP) ID.
__device__ uint32_t
smid_get()
{
  uint smid = 0;
  asm( "mov.u32 %0, %%smid;" : "=r" (smid) );
  return smid;
}

__global__ void 
flexspmm_cuda_k4_vec1_v31(int* mdseg_rowPtr_dev, float* mdsegNzCV_dev, int*  mdsegVoMap_dev,
                                int* mdgrouped_tailSeg_dev, int* mdnext_seg_dev, 
                                int mdm, int mdn, int mdk, int mdsms, int mdn_segs,
                                float* mdshadow_b_dev, float* mdmat_c_dev){ 
    // requires preprocess dense mat B
    
    const int wp_id = threadIdx.x>>5; //threadIdx.x / 32;
    const int lane_id = threadIdx.x & 0x1f; //threadIdx.x % 32;
    //const int th_p_row = 4;
    const int row_id = lane_id>>2; //lane_id / th_p_row; // 4 threads process a row
    int c_col = lane_id & 0x3; //lane_id % th_p_row;


    __shared__ int seg_idx[2];
    uint32_t sm_id = smid_get();

    
    int gold_row_id[tm];
    
    int nsi = sm_id;
    // the sentinel tile-seg of the nsi-th SM
    const int tail_seg_idx = mdgrouped_tailSeg_dev[nsi];
    while ( true ) {

        int seg_idx_0 = lane_id ? 0 : atomicAdd( &mdnext_seg_dev[ nsi ], 1 );

        __syncwarp();
        if ( lane_id == 0 ) seg_idx[ wp_id ] = seg_idx_0;
        __syncwarp();

        if ( nsi < mdsms && seg_idx[ wp_id ] >= tail_seg_idx ) { nsi = mdsms; continue; }
        if ( nsi == mdsms && seg_idx[ wp_id ] >= mdn_segs ) break;
    
 
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = mdsegVoMap_dev[seg_idx[ wp_id ]*tm+i];
        }
       
        // visit all rows of the segment
        // step, 32 / th_p_row
        for ( int r_idx = row_id; r_idx<tm; r_idx += 8 ){
            // the global idx of the first non-zero of this tile-seg 
            //int seg_cur_id = mdsegPtr_dev[ seg_idx[ wp_id ] ];

            int cur_rowPtr = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx ]; 
            int nnz_cur_row = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx + 1 ] - cur_rowPtr;
            int atomicORnot = gold_row_id[ r_idx ] & (1<<31); // get MSB
            int actual_row = gold_row_id[ r_idx ] & 0x7fffffff;
            int addr = actual_row*mdk;
            
            for (int cc=c_col; cc<mdk; cc+=4){
                float res = 0;
                for ( int z=0; z<nnz_cur_row; z += 1 ){ // over non-zeros of the row
                 
                   // column & val 
                   float2 cv = reinterpret_cast<float2*>(mdsegNzCV_dev)[ cur_rowPtr + z ];
                   
                   float bv = mdshadow_b_dev[ (int)cv.x*mdk + cc ];
                   res += cv.y * bv; 
                   
                }
                
                
                // store results back  
                if ( actual_row<mdm ){
                    
                    if ( atomicORnot>>31 ){
                        atomicAdd( &mdmat_c_dev[ addr + cc ], res);
                    }else{
                        mdmat_c_dev[ addr + cc ] = res;
                    }
                }
            } 
        } // end tile-seg row loop
    } // end tile-segs loops
}

__global__ void 
flexspmm_cuda_k8_vec2_v32(int* mdseg_rowPtr_dev, float* mdsegNzCV_dev, int*  mdsegVoMap_dev,
                            int* mdgrouped_tailSeg_dev, int* mdnext_seg_dev, 
                            int mdm, int mdn, int mdk, int mdsms, int mdn_segs,
                            float* mdshadow_b_dev, float* mdmat_c_dev){ 
    // requires preprocess dense mat B
    
    const int wp_id = threadIdx.x>>5; //threadIdx.x / 32;
    const int lane_id = threadIdx.x & 0x1f; //threadIdx.x % 32;
    //const int th_p_row = 4;
    const int row_id = lane_id>>2; //lane_id / th_p_row; // 4 threads process a row
    int c_col = lane_id & 0x3; //lane_id % th_p_row;


    __shared__ int seg_idx[2];
    uint32_t sm_id = smid_get();

    
    int gold_row_id[tm];
    
    int nsi = sm_id;
    // the sentinel tile-seg of the nsi-th SM
    const int tail_seg_idx = mdgrouped_tailSeg_dev[nsi];
    while ( true ) {

        int seg_idx_0 = lane_id ? 0 : atomicAdd( &mdnext_seg_dev[ nsi ], 1 );

        __syncwarp();
        if ( lane_id == 0 ) seg_idx[ wp_id ] = seg_idx_0;
        __syncwarp();

        if ( nsi < mdsms && seg_idx[ wp_id ] >= tail_seg_idx ) { nsi = mdsms; continue; }
        if ( nsi == mdsms && seg_idx[ wp_id ] >= mdn_segs ) break;
 
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = mdsegVoMap_dev[seg_idx[ wp_id ]*tm+i];
        }
       
        // visit all rows of the segment
        // step, 32 / th_p_row
        for ( int r_idx = row_id; r_idx<tm; r_idx += 8 ){
            // the global idx of the first non-zero of this tile-seg 
            //int seg_cur_id = mdsegPtr_dev[ seg_idx[ wp_id ] ];

            
            int cur_rowPtr = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx ]; 
            int nnz_cur_row = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx + 1 ] - cur_rowPtr;
            int actual_row = gold_row_id[ r_idx ] & 0x7fffffff;
            int atomicORnot = gold_row_id[ r_idx ] & (1<<31); // get MSB
            int addr = actual_row*mdk;
            
            for (int cc=c_col; cc*2<mdk; cc+=8){
                float res[2]{};
                for ( int z=0; z<nnz_cur_row; z += 1 ){ // over non-zeros of the row
                 
                   // column & val 
                   float2 cv = reinterpret_cast<float2*>(mdsegNzCV_dev)[ cur_rowPtr + z ];
                   
                   float *shadow_b_addr = &mdshadow_b_dev[ (int)cv.x*mdk ];
                   
                   float2 b_vec = reinterpret_cast<float2*>(shadow_b_addr)[ cc ];
                   
                   res[0] += cv.y * b_vec.x; 
                   if ( cc*2+1 < mdk ){
                        res[1] += cv.y * b_vec.y; 
                   }
                   
                }
                    
                // store results back  
                if ( actual_row<mdm ){
                    
                    if ( atomicORnot>>31 ){
                        atomicAdd( &mdmat_c_dev[ addr + cc*2 ], res[0]);
                        if ( cc*2+1 < mdk )
                            atomicAdd( &mdmat_c_dev[ addr + cc*2 + 1 ], res[1]);
                    }else{
                        mdmat_c_dev[ addr + cc*2 ] = res[0];
                        if ( cc*2+1 < mdk ){
                            mdmat_c_dev[ addr + cc*2 + 1 ] = res[1];
                        }
                    }
                }
            } 
        } // end tile-seg row loop
    } // end tile-segs loops
}

__global__ void 
flexspmm_cuda_k16_vec4_v33(int* mdseg_rowPtr_dev, float* mdsegNzCV_dev, int*  mdsegVoMap_dev,
                            int* mdgrouped_tailSeg_dev, int* mdnext_seg_dev, 
                            int mdm, int mdn, int mdk, int mdsms, int mdn_segs,
                            float* mdshadow_b_dev, float* mdmat_c_dev){ 
    // requires preprocess dense mat B
    
    const int wp_id = threadIdx.x>>5; //threadIdx.x / 32;
    const int lane_id = threadIdx.x & 0x1f; //threadIdx.x % 32;
    //const int th_p_row = 4;
    const int row_id = lane_id>>2; //lane_id / th_p_row; // 4 threads process a row
    int c_col = lane_id & 0x3; //lane_id % th_p_row;


    __shared__ int seg_idx[2];
    uint32_t sm_id = smid_get();

    
    int gold_row_id[tm];
    
    int nsi = sm_id;
    // the sentinel tile-seg of the nsi-th SM
    const int tail_seg_idx = mdgrouped_tailSeg_dev[nsi];
    while ( true ) {

        int seg_idx_0 = lane_id ? 0 : atomicAdd( &mdnext_seg_dev[ nsi ], 1 );

        __syncwarp();
        if ( lane_id == 0 ) seg_idx[ wp_id ] = seg_idx_0;
        __syncwarp();

        if ( nsi < mdsms && seg_idx[ wp_id ] >= tail_seg_idx ) { nsi = mdsms; continue; }
        if ( nsi == mdsms && seg_idx[ wp_id ] >= mdn_segs ) break;
 
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = mdsegVoMap_dev[seg_idx[ wp_id ]*tm+i];
        }
       
        // visit all rows of the segment
        // step, 32 / th_p_row
        for ( int r_idx = row_id; r_idx<tm; r_idx += 8 ){
            // the global idx of the first non-zero of this tile-seg 
            //int seg_cur_id = mdsegPtr_dev[ seg_idx[ wp_id ] ];

            
            int cur_rowPtr = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx ]; 
            int nnz_cur_row = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx + 1 ] - cur_rowPtr;
            
            for (int cc=c_col; cc*4<mdk; cc+=16){
                float res[4]{};
                for ( int z=0; z<nnz_cur_row; z += 1 ){ // over non-zeros of the row
                 
                   // column & val 
                   float2 cv = reinterpret_cast<float2*>(mdsegNzCV_dev)[ cur_rowPtr + z ];
                   
                   float *shadow_b_addr = &mdshadow_b_dev[ (int)cv.x*mdk ];
                   
                   float4 b_vec = reinterpret_cast<float4*>(shadow_b_addr)[ cc ];
                   
                   res[0] += cv.y * b_vec.x; 
                   if ( cc*4+3 < mdk ){
                        res[1] += cv.y * b_vec.y; 
                        res[2] += cv.y * b_vec.z; 
                        res[3] += cv.y * b_vec.w; 
                   }else if ( cc*4+2 < mdk ){
                        res[1] += cv.y * b_vec.y; 
                        res[2] += cv.y * b_vec.z; 
                   }else if ( cc*4+1 < mdk ){
                        res[1] += cv.y * b_vec.y; 
                   } 
                  
                }
                
                int actual_row = gold_row_id[ r_idx ] & 0x7fffffff;
                
                // store results back  
                if ( actual_row<mdm ){
                    int atomicORnot = gold_row_id[ r_idx ] & (1<<31); // get MSB
                    int addr = actual_row*mdk;
                    
                    if ( atomicORnot>>31 ){
                        atomicAdd( &mdmat_c_dev[ addr + cc*4 ], res[0]);
                        if ( cc*4+3 < mdk ){
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 1 ], res[1]);
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 2 ], res[2]);
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 3 ], res[3]);
                        }else if ( cc*4+2 < mdk ){
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 1 ], res[1]);
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 2 ], res[2]);
                        }else if ( cc*4+1 < mdk ){
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 1 ], res[1]);
                        }
                    }else{
                        if ( cc*4+3 < mdk ){
                            //md.mat_c_dev[ addr + c_col*4 ] = res[0];
                            //md.mat_c_dev[ addr + c_col*4 + 1 ] = res[1];
                            //md.mat_c_dev[ addr + c_col*4 + 2 ] = res[2];
                            //md.mat_c_dev[ addr + c_col*4 + 3 ] = res[3];
                            float* mat_c = &mdmat_c_dev[ addr ];
                            float4 vect4_c = {res[0], res[1], res[2], res[3]};
                            reinterpret_cast<float4*>(mat_c)[ cc ] = vect4_c;
                        }else if ( cc*4+2 < mdk ){
                            mdmat_c_dev[ addr + cc*4 ] = res[0];
                            mdmat_c_dev[ addr + cc*4 + 1 ] = res[1];
                            mdmat_c_dev[ addr + cc*4 + 2 ] = res[2];
                        }else if ( cc*4+1 < mdk ){
                            mdmat_c_dev[ addr + cc*4 ] = res[0];
                            mdmat_c_dev[ addr + cc*4 + 1 ] = res[1];
                        }else{
                            mdmat_c_dev[ addr + cc*4 ] = res[0];
                        }
                    }
                }
            } 
        } // end tile-seg row loop
    } // end tile-segs loops
    
}

__global__ void 
flexspmm_cuda_k32_vec4_v34(int* mdseg_rowPtr_dev, float* mdsegNzCV_dev, int*  mdsegVoMap_dev,
                            int* mdgrouped_tailSeg_dev, int* mdnext_seg_dev, 
                            int mdm, int mdn, int mdk, int mdsms, int mdn_segs,
                            float* mdshadow_b_dev, float* mdmat_c_dev){ 
    // requires preprocess dense mat B
    
    const int wp_id = threadIdx.x>>5; //threadIdx.x / 32;
    const int lane_id = threadIdx.x & 0x1f; //threadIdx.x % 32;
    //const int th_p_row = 8;
    const int row_id = lane_id>>3; //lane_id / th_p_row; // 8 threads process a row
    int c_col = lane_id & 0x7; //lane_id % th_p_row;


    __shared__ int seg_idx[2];
    uint32_t sm_id = smid_get();

    
    int gold_row_id[tm];
    
    int nsi = sm_id;
    // the sentinel tile-seg of the nsi-th SM
    const int tail_seg_idx = mdgrouped_tailSeg_dev[nsi];
    while ( true ) {

        int seg_idx_0 = lane_id ? 0 : atomicAdd( &mdnext_seg_dev[ nsi ], 1 );

        __syncwarp();
        if ( lane_id == 0 ) seg_idx[ wp_id ] = seg_idx_0;
        __syncwarp();

        if ( nsi < mdsms && seg_idx[ wp_id ] >= tail_seg_idx ) { nsi = mdsms; continue; }
        if ( nsi == mdsms && seg_idx[ wp_id ] >= mdn_segs ) break;
 
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = mdsegVoMap_dev[seg_idx[ wp_id ]*tm+i];
        }
       
        // visit all rows of the segment
        // step, 32 / th_p_row
        for ( int r_idx = row_id; r_idx<tm; r_idx += 4 ){
            // the global idx of the first non-zero of this tile-seg 
            //int seg_cur_id = mdsegPtr_dev[ seg_idx[ wp_id ] ];

            
            int cur_rowPtr = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx ]; 
            int nnz_cur_row = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx + 1 ] - cur_rowPtr;
            
            for (int cc=c_col; cc*4<mdk; cc+=32){
                float res[4]{};
                for ( int z=0; z<nnz_cur_row; z += 1 ){ // over non-zeros of the row
                 
                   // column & val 
                   float2 cv = reinterpret_cast<float2*>(mdsegNzCV_dev)[ cur_rowPtr + z ];
                   
                   float *shadow_b_addr = &mdshadow_b_dev[ (int)cv.x*mdk ];
                   
                   float4 b_vec = reinterpret_cast<float4*>(shadow_b_addr)[ cc ];
                   
                   res[0] += cv.y * b_vec.x; 
                   if ( cc*4+3 < mdk ){
                        res[1] += cv.y * b_vec.y; 
                        res[2] += cv.y * b_vec.z; 
                        res[3] += cv.y * b_vec.w; 
                   }else if ( cc*4+2 < mdk ){
                        res[1] += cv.y * b_vec.y; 
                        res[2] += cv.y * b_vec.z; 
                   }else if ( cc*4+1 < mdk ){
                        res[1] += cv.y * b_vec.y; 
                   } 
                  
                }
                
                int actual_row = gold_row_id[ r_idx ] & 0x7fffffff;
                
                // store results back  
                if ( actual_row<mdm ){
                    int atomicORnot = gold_row_id[ r_idx ] & (1<<31); // get MSB
                    int addr = actual_row*mdk;
                    
                    if ( atomicORnot>>31 ){
                        atomicAdd( &mdmat_c_dev[ addr + cc*4 ], res[0]);
                        if ( cc*4+3 < mdk ){
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 1 ], res[1]);
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 2 ], res[2]);
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 3 ], res[3]);
                        }else if ( cc*4+2 < mdk ){
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 1 ], res[1]);
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 2 ], res[2]);
                        }else if ( cc*4+1 < mdk ){
                            atomicAdd( &mdmat_c_dev[ addr + cc*4 + 1 ], res[1]);
                        }
                    }else{
                        if ( cc*4+3 < mdk ){
                            //md.mat_c_dev[ addr + c_col*4 ] = res[0];
                            //md.mat_c_dev[ addr + c_col*4 + 1 ] = res[1];
                            //md.mat_c_dev[ addr + c_col*4 + 2 ] = res[2];
                            //md.mat_c_dev[ addr + c_col*4 + 3 ] = res[3];
                            float* mat_c = &mdmat_c_dev[ addr ];
                            float4 vect4_c = {res[0], res[1], res[2], res[3]};
                            reinterpret_cast<float4*>(mat_c)[ cc ] = vect4_c;
                        }else if ( cc*4+2 < mdk ){
                            mdmat_c_dev[ addr + cc*4 ] = res[0];
                            mdmat_c_dev[ addr + cc*4 + 1 ] = res[1];
                            mdmat_c_dev[ addr + cc*4 + 2 ] = res[2];
                        }else if ( cc*4+1 < mdk ){
                            mdmat_c_dev[ addr + cc*4 ] = res[0];
                            mdmat_c_dev[ addr + cc*4 + 1 ] = res[1];
                        }else{
                            mdmat_c_dev[ addr + cc*4 ] = res[0];
                        }
                    }
                }
            } 
        } // end tile-seg row loop
    } // end tile-segs loops
}

__global__ void 
flexspmm_cuda_vec1_v35(int* mdseg_rowPtr_dev, float* mdsegNzCV_dev, int*  mdsegVoMap_dev,
                                int* mdgrouped_tailSeg_dev, int* mdnext_seg_dev, 
                                int mdm, int mdn, int mdk, int mdsms, int mdn_segs,
                                float* mdshadow_b_dev, float* mdmat_c_dev){ 
    // requires preprocess dense mat B
    
    const int wp_id = threadIdx.x>>5; //threadIdx.x / 32;
    const int lane_id = threadIdx.x & 0x1f; //threadIdx.x % 32;
    //const int th_p_row = 4;
    const int row_id = lane_id>>3; //lane_id / th_p_row; // 8 threads process a row
    int c_col = lane_id & 0x7; //lane_id % th_p_row;


    __shared__ int seg_idx[2];
    uint32_t sm_id = smid_get();
 
    int gold_row_id[tm];
    
    int nsi = sm_id;
    // the sentinel tile-seg of the nsi-th SM
    const int tail_seg_idx = mdgrouped_tailSeg_dev[nsi];
    while ( true ) {

        int seg_idx_0 = lane_id ? 0 : atomicAdd( &mdnext_seg_dev[ nsi ], 1 );

        __syncwarp();
        if ( lane_id == 0 ) seg_idx[ wp_id ] = seg_idx_0;
        __syncwarp();

        if ( nsi < mdsms && seg_idx[ wp_id ] >= tail_seg_idx ) { nsi = mdsms; continue; }
        if ( nsi == mdsms && seg_idx[ wp_id ] >= mdn_segs ) break;
 
        #pragma unroll
        for (int i=0; i<tm; ++i){
            gold_row_id[i] = mdsegVoMap_dev[seg_idx[ wp_id ]*tm+i];
        }
       
        // visit all rows of the segment
        // step, 32 / th_p_row
        for ( int r_idx = row_id; r_idx<tm; r_idx += 4 ){
            // the global idx of the first non-zero of this tile-seg 
            //int seg_cur_id = mdsegPtr_dev[ seg_idx[ wp_id ] ];

            int cur_rowPtr = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx ]; 
            int nnz_cur_row = mdseg_rowPtr_dev[ seg_idx[ wp_id ]*(tm+1) + r_idx + 1 ] - cur_rowPtr;
            int atomicORnot = gold_row_id[ r_idx ] & (1<<31); // get MSB
            int actual_row = gold_row_id[ r_idx ] & 0x7fffffff;
            int addr = actual_row*mdk;
            
            for (int cc=c_col; cc<mdk; cc+=8){
                float res = 0;
                for ( int z=0; z<nnz_cur_row; z += 1 ){ // over non-zeros of the row
                 
                   // column & val 
                   float2 cv = reinterpret_cast<float2*>(mdsegNzCV_dev)[ cur_rowPtr + z ];
                   
                   float bv = mdshadow_b_dev[ (int)cv.x*mdk + cc ];
                   res += cv.y * bv; 
                   
                }
                 
                // store results back  
                if ( actual_row<mdm ){
                    
                    if ( atomicORnot>>31 ){
                        atomicAdd( &mdmat_c_dev[ addr + cc ], res);
                    }else{
                        mdmat_c_dev[ addr + cc ] = res;
                    }
                }
            } 
        } // end tile-seg row loop
    } // end tile-segs loops
}
void flexspmm(int* mdseg_rowPtr_dev, float* mdsegNzCV_dev, int*  mdsegVoMap_dev,
              int* mdgrouped_tailSeg_dev, int* mdnext_seg_dev, 
              int mdm, int mdn, int mdk, int mdn_segs,
              float* mdshadow_b_dev, float* mdmat_c_dev){
    
    //printf("%d of %s: m = %d, n = %d, k = %d, n_segs = %d\n",__LINE__,__FILE__,
    //        mdm,mdn,mdk,mdn_segs);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); 
    int mdsms = deviceProp.multiProcessorCount;

    if (mdk==8){
        //printf("\n%d of %s: kernel V32\n", __LINE__,__FILE__);
        flexspmm_cuda_k8_vec2_v32<<<32*mdsms,64>>>(mdseg_rowPtr_dev, mdsegNzCV_dev, mdsegVoMap_dev,
                                  mdgrouped_tailSeg_dev, mdnext_seg_dev, 
                                  mdm, mdn, mdk, mdsms, mdn_segs,
                                  mdshadow_b_dev, mdmat_c_dev);
    }else if (mdk==16){
        //printf("\n%d of %s: kernel V33\n", __LINE__,__FILE__);
        flexspmm_cuda_k16_vec4_v33<<<32*mdsms,64>>>(mdseg_rowPtr_dev, mdsegNzCV_dev, mdsegVoMap_dev,
                                   mdgrouped_tailSeg_dev, mdnext_seg_dev, 
                                   mdm, mdn, mdk, mdsms, mdn_segs,
                                   mdshadow_b_dev, mdmat_c_dev);
    }else if (mdk==32){
        //printf("\n%d of %s: kernel V34\n", __LINE__,__FILE__);
        flexspmm_cuda_k32_vec4_v34<<<32*mdsms,64>>>(mdseg_rowPtr_dev, mdsegNzCV_dev, mdsegVoMap_dev,
                                   mdgrouped_tailSeg_dev, mdnext_seg_dev, 
                                   mdm, mdn, mdk, mdsms, mdn_segs,
                                   mdshadow_b_dev, mdmat_c_dev);
    }else if (mdk<32){
        //printf("\n%d of %s: kernel V31\n", __LINE__,__FILE__);
        flexspmm_cuda_k4_vec1_v31<<<32*mdsms,64>>>(mdseg_rowPtr_dev, mdsegNzCV_dev, mdsegVoMap_dev,
                                  mdgrouped_tailSeg_dev, mdnext_seg_dev, 
                                  mdm, mdn, mdk, mdsms, mdn_segs,
                                  mdshadow_b_dev, mdmat_c_dev); 
    }else{
        // TO DO
        //printf("\n%d of %s: kernel V35\n", __LINE__,__FILE__);
        flexspmm_cuda_vec1_v35<<<32*mdsms,64>>>(mdseg_rowPtr_dev, mdsegNzCV_dev, mdsegVoMap_dev,
                                  mdgrouped_tailSeg_dev, mdnext_seg_dev, 
                                  mdm, mdn, mdk, mdsms, mdn_segs,
                                  mdshadow_b_dev, mdmat_c_dev); 
    }
    
    return ;
}
}
