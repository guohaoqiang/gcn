#include "../include/flex.h"

void convert(DataLoader& input){
    mat data(input.cpuA->row, input.cpuA->col, input.cpuA->vals, input.cpuA->r, input.cpuA->c, input.cpuA->nnz);
	data.csr2tile();
	data.print1();
	//data.print2();
}
