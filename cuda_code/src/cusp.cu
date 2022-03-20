#include "../include/cusp.cuh"

int run1(DataLoader& input, Metrics& metric){
    T *gpuB = nullptr; // n * c
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gpuB), sizeof(T) * input.n * input.c));
    float duration;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // ############################
    cudaEventRecord(start);
    // ############################

    //----------  B = XW : sgemm------------
    const float alpha = 1.0;
    const float beta = 0.0;
    
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    /* step 2: compute */
    CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, input.n, input.c, input.dim, &alpha, 
                input.gpuX, input.n, input.gpuW, input.dim, &beta, gpuB, input.n));
    CUBLAS_CHECK(cublasDestroy(cublasH));
    LOG(INFO) << "step1 of run1 completed ...";

    //----------  C = AB : sparsemm------------
    // CUSPARSE APIs
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
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, input.n, input.c, input.n, gpuB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, input.n, input.c, input.n, input.gpuRef1,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    LOG(INFO) << "step2 of run1 completed ...";
    // ############################
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// ############################
	cudaEventElapsedTime(&duration, start, stop);
    metric.t += duration;
    metric.flops = (input.cpuA->nnz * input.c + input.n * input.dim * input.c) * 2;
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
    CUDA_CHECK( cudaFree(gpuB) );

    CUDA_CHECK(cudaMemcpy(&(input.cpuRef1[0]), input.gpuRef1, sizeof(T)*input.n*input.c, cudaMemcpyDeviceToHost));
    LOG(INFO) << "run1 completed ...";
    return 0;
}

int run2(DataLoader& input, Metrics& metric){
    T *gpuB = nullptr; // n * dim
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&gpuB), sizeof(T) * input.n * input.dim));
    float duration;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // ############################
    cudaEventRecord(start);
    // ############################

    //----------  B = AX : sparsemm------------
    const float alpha = 1.0;
    const float beta = 0.0;
    // CUSPARSE APIs
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
    // Create dense matrix X
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, input.n, input.dim, input.n, input.gpuX,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, input.n, input.dim, input.n, gpuB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
    LOG(INFO) << "step1 of run2 completed ...";
    //----------  C = BW : sgemm------------
    
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    cublasHandle_t cublasH = NULL;
    CUBLAS_CHECK(cublasCreate(&cublasH));

    /* step 2: compute */
    CUBLAS_CHECK(cublasSgemm(cublasH, transa, transb, input.n, input.c, input.dim, &alpha, 
                gpuB, input.n, input.gpuW, input.dim, &beta, input.gpuRef2, input.n));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    LOG(INFO) << "step2 of run2 completed ...";

    // ############################
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	// ############################
	cudaEventElapsedTime(&duration, start, stop);
    metric.t += duration;
    metric.flops = (input.cpuA->nnz * input.dim + input.n * input.dim * input.c) * 2;
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    CHECK_CUDA( cudaFree(dBuffer) )
    CUDA_CHECK( cudaFree(gpuB) );

    CUDA_CHECK(cudaMemcpy(&(input.cpuRef2[0]), input.gpuRef2, sizeof(T)*input.n*input.c, cudaMemcpyDeviceToHost));
    LOG(INFO) << "run2 completed ...";
    return 0;
}

