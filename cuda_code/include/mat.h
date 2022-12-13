#ifndef MAT_H
#define MAT_H 
#include <vector>
#include <iostream>
#include <algorithm>
#define DataType float
#define DEBUG
using namespace std;
class mat{
public:
	int m,k;
    int n;
	int nnz;
	int pos;
	int tm,tn;
	std::vector<unsigned int>& rowPtr;
	std::vector<unsigned int>& colIdx;
	std::vector<DataType>& vals;

	mat(std::vector<unsigned int>& rowPtr, std::vector<unsigned int>& colIdx, std::vector<DataType>& vals, int h, int w, int n);
	void print1();
	void print2();
	void csr2tile();

	std::vector<unsigned int> tileRowPtr;
	std::vector<unsigned int> nnzPtr;
	std::vector<unsigned int> tileLeftColIdx;
    std::vector<char> rc_Offset;
	std::vector<DataType> newVals;
    
    std::vector<unsigned int> rowOffset;
	std::vector<unsigned int> tileColIdx;

	// regular sparse-tile storage
	std::vector<unsigned int> rgl_tileRowPtr;
	std::vector<unsigned int> rgl_tileColIdx;
	std::vector<unsigned int> rgl_nnzPtr;
	std::vector<unsigned int> rgl_rowOffset;
	std::vector<unsigned int> rgl_colOffset;
	std::vector<DataType> rgl_newVals;

	void csr2regular(int i);
	void csr2flex(int i);
};
#endif /* MAT_H */
