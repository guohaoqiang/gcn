#include "../include/main.h"
#define T float
DEFINE_string(input, "../data/cora.csv", "The name of benchmarks.");
DEFINE_int32(dim, 32, "The dims of output.");
DEFINE_bool(cmp, true, "Compare results");

int main(int argc, char *argv[])
{
    int WarmupIterations  = 5;
    int ExecutionIterations = 10;
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    VLOG(2)<<"Loading bench config....";
    DataLoader data(FLAGS_input, FLAGS_dim);
    std::cout<<"Graph name: "<<data.graph_name<<std::endl;
    std::cout<<"A: "<<data.cpuA->r<<"*"<<data.cpuA->c<<"  X: "<<data.n<<"*"<<data.dim<<"   W: "<<data.dim<<"*"<<data.c<<std::endl;
    std::cout<<"NNZ of A: "<<data.cpuA->nnz<<std::endl;
    Metrics baselinemetrics1, baselinemetrics2, benchmetrics;
    if (FLAGS_cmp){
        for (size_t i=0; i<WarmupIterations; ++i){
            Metrics metric0 = Metrics();
            run1(data,metric0);
            //CUDA_CHECK(cudaMemset(data.gpuC, 0, sizeof(T) * data.n * data.c));
            run2(data,metric0);
        } 
        for (size_t i=0; i<ExecutionIterations; ++i){
            // step1: B = XW
            // step2: C = AB
            Metrics metric1 = Metrics();
            run1(data,metric1);
            baselinemetrics1 += metric1;
           
            //CUDA_CHECK(cudaMemset(data.gpuC, 0, sizeof(T) * data.n * data.c));
            // step1: B = AX
            // step2: C = BW 
            Metrics metric2 = Metrics();
            run2(data,metric2);
            baselinemetrics2 += metric2;
        } 
    } 

    for (size_t i=0; i<WarmupIterations; ++i){
        Metrics metric = Metrics();
        run();
    } 
    for (size_t i=0; i<ExecutionIterations; ++i){
        Metrics metric = Metrics();
        run();
        benchmetrics += metric;
    } 

    //std::cout<<"A(XW): "<< baselinemetrics1.flops/baselinemetrics1.t << "Gflops/s"<<std::endl;
    //std::cout<<"(AX)W: "<< baselinemetrics2.flops/baselinemetrics2.t << "Gflops/s"<<std::endl;
    //std::cout<<"AXW: "<< benchmetrics.flops/benchmetrics.t << "Gflops/s"<<std::endl;
    
    return 0;

}

