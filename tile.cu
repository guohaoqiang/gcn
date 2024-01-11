#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <map>
#include <ranges>
#include <bit>

extern "C"{

void csr2seg_Cmajor(int ridx, int* rowPtr, int* colIdx, float* vals, int m, int n, int nnz,
              int* voMp, int* segVoMap, int* seg_rowPtr, float* segNzCV, int tm, int* n_segs){
	// row tile upper bound and lower bound
	int rowStart = ridx * tm;
	int rowEnd = min(m, (ridx+1)*tm); // exclusive

	// keep track of the cols in each row
	std::vector<int> cOffset(tm, 0);

    // {col, val}, for kernel v31 
    std::vector<std::vector<std::pair<int,float>>> segcv(tm, std::vector<std::pair<int,float>>()); 

    int nnz_limit = 128;
    int dif = 0.1*nnz_limit; 
    int nnzInSeg = 0;

    // If n_nodes_z_out>0 some panels can be empty, which tiling can't handle.
    //assert( !dl.dl_original->n_nodes_z_out );
    std::vector<int> atom(tm, 0);

    std::map<int,int> occ_cols;
    for ( auto c: std::views::iota(rowPtr[rowStart],rowPtr[rowEnd]) )
      occ_cols[colIdx[c]]++;
    const auto last_col = occ_cols.rbegin()->first;
    // collect segs in the panel
    for ( auto [j,ncol]: occ_cols ) {
        
        for ( int i=rowStart; i<rowEnd; ++i ){
            // absolute position of the nze in csr, idx = base + offset
            int c = rowPtr[i] + cOffset[i-rowStart];
            if ( colIdx[c]==j && c<rowPtr[i+1] ){
                // nze values
                
                segcv[i-rowStart].push_back({j,vals[c]}); // for v31 kernel

                cOffset[i-rowStart]++;
                atom[i-rowStart]++;
                nnzInSeg++;
            }
        }
        if ( (j==last_col && nnzInSeg) || (nnz_limit - nnzInSeg)<=dif || nnzInSeg>nnz_limit ){
        
            // for kernel v31
            //if ( !seg_rowPtr.empty() ) seg_rowPtr.push_back( seg_rowPtr.back() + 0 );
            //else seg_rowPtr.push_back( 0 );
            if ( n_segs[0] ) 
                seg_rowPtr[ (tm+1)*n_segs[0] ] = seg_rowPtr[ (tm+1)*n_segs[0]-1 ];
            else seg_rowPtr[ (tm+1)*n_segs[0] ] = 0;
           
            int nz_idx_in_seg = 0; 
            int seg_row_begin = seg_rowPtr[ (tm+1)*n_segs[0] ];
            for ( int i=0; i<tm; ++i ){
                //seg_rowPtr.push_back( seg_rowPtr.back() + segcv[i].size() );
                for ( auto &p:segcv[i] ){
                    //segNzCV.push_back((float)p.first); // col of the nz is stored in float
                    //segNzCV.push_back(p.second);
                    segNzCV[ (seg_row_begin + nz_idx_in_seg)*2 ] = (float)p.first; // col of the nz is stored in float
                    segNzCV[ (seg_row_begin + nz_idx_in_seg)*2 + 1 ] = p.second;
                    nz_idx_in_seg++;
                }
                seg_rowPtr[ (tm+1)*n_segs[0]+i+1 ] = seg_row_begin + segcv[i].size();
                segcv[i].clear();
            }
            //segPtr.push_back(segPtr.back()+nnzInSeg);
            nnzInSeg = 0;
           
            for (int i=rowStart; i<rowStart+tm; ++i){
                if ( i<rowEnd ){
                    if ( atom[i-rowStart]>=0 && atom[i-rowStart]<(rowPtr[i+1]-rowPtr[i]) ){
                        // if the #nz in a specific row of a seg 
                        // is less than that of the whole row,
                        // the row requires "atomic add".
                        // use MSB to mark it.
                        //segVoMap.push_back( voMp[i] | (1<<31) );
                        //segVoMap[ tm*n_segs[0]+i-rowStart ] = ( voMp[i] | (1<<31) );
                        segVoMap[ tm*n_segs[0]+i-rowStart ] = ( i | (1<<31) );
                    }else{ 
                        //segVoMap.push_back( voMp[i] );
                        //segVoMap[ tm*n_segs[0]+i-rowStart ] = voMp[i];
                        segVoMap[ tm*n_segs[0]+i-rowStart ] = i;
                    }
                }else{
                    // for the last panel, the rows may be less than tm 
                    //segVoMap.push_back(1<<(bit_width((uint)m)+1));
                    segVoMap[ tm*n_segs[0]+i-rowStart ] = (1<<(std::bit_width((uint)m)+1));
                }
                
                atom[ i-rowStart ] = 0;
            }
            n_segs[0]++;
        }
    }
}
void csr2tile(int* rowPtr, int* colIdx, float* vals, int m, int n, int nnz,
              int* vo_mp, int* segVoMap, int* seg_rowPtr, float* segNzCV,
              int* grouped_tailSeg, int* next_seg, int tm, int* n_segs){
	
    bool print_bucket = false;
	int tileRows = (m+tm-1)/tm;
	for (int i=0; i<tileRows; ++i){
        csr2seg_Cmajor(i, rowPtr, colIdx, vals, m, n, nnz,
                vo_mp, segVoMap, seg_rowPtr, segNzCV, tm, n_segs);
	} 
    //n_segs = segPtr.size()-1;
    if (print_bucket) 
        printf("%d of %s, n_segs = %d\n",__LINE__, __FILE__, n_segs[0]); 
    
        int device_id;
        cudaDeviceProp prop;
        cudaGetDevice( &device_id );
        cudaGetDeviceProperties( &prop, device_id );
        int n_sm = prop.multiProcessorCount;
        
        // distribute segs into n_sm+1 buckets, contiguous segs are in a bucket
        // according to #non zeros ( wkload per sm )
        // to balance workload, the last bucket is to offer segs when faster SMs are free   
        //int nnz = newVals.size(); 
        int wkload = nnz / n_sm; 
        int seg_head_sm = 0;
        int seg_tail_sm;
        int validate_nnz = 0;

        // assign segs to each sm bucket
        for (int i=0; i<n_sm; ++i){
            //next_seg.push_back( seg_head_sm );
            next_seg[i] = seg_head_sm;
            //int nz = segPtr[seg_head_sm+1] - segPtr[seg_head_sm];
            int nz = seg_rowPtr[(seg_head_sm+1)*(tm+1)] - seg_rowPtr[seg_head_sm*(tm+1)];
            
            seg_tail_sm = seg_head_sm + 1;
            while ( seg_tail_sm < n_segs[0] && nz<(int)(0.95*wkload) ){
                //nz += (segPtr[seg_tail_sm+1] - segPtr[seg_tail_sm]);
                nz += (seg_rowPtr[(seg_tail_sm+1)*(tm+1)] - seg_rowPtr[seg_tail_sm*(tm+1)]);
                seg_tail_sm++;
            }
            validate_nnz += nz;
            //grouped_tailSeg.push_back( min(n_segs,seg_tail_sm) );
            grouped_tailSeg[i] = min(n_segs[0],seg_tail_sm);
            if (print_bucket) printf("%d of %s, %d#sm: start = %d, end = %d\n", 
                    __LINE__,__FILE__, i,seg_head_sm,grouped_tailSeg[i]);
            seg_head_sm = seg_tail_sm;
        }
        if (false) 
            printf("%d of %s, normal_nnz = %d\n", __LINE__,__FILE__,validate_nnz);
        
        // the last bucket is used for workload balance among SMs 
        // if seg_head_sm==n_segs, then n_segs==seg_head_sm
        //next_seg.push_back( seg_head_sm );
        next_seg[n_sm] = seg_head_sm;
        //grouped_tailSeg.push_back( n_segs );
        grouped_tailSeg[n_sm] = n_segs[0];
        if (print_bucket) 
            printf("%d of %s, %d#sm: start = %d, end = %d\n", 
                    __LINE__,__FILE__, n_sm, seg_head_sm,grouped_tailSeg[n_sm]);
        //validate_nnz += segPtr[n_segs]-segPtr[seg_head_sm];
        //assert( validate_nnz==segPtr.back() );
        //assert( grouped_tailSeg.size()==n_sm+1 );
        //assert( next_seg.size()==n_sm+1 );    
}
}
