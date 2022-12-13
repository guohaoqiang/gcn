#ifndef FLEX_H
#define FLEX_H 
#include "DataLoader.cuh"
#include "mat.h"
#include "flex_spmm.cuh"

void convert(DataLoader& input);


void flexspgemm(mat& data);


#endif /* FLEX_H */
