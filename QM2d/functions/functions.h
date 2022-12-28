#pragma once 

#include "../params.h"
#include "../SpinorField/SpinorField.h"
#include <complex>
#include <iostream>

typedef std::vector<std::vector<std::complex<double>>> mat;
class SpinorField;

/*
    Solve Ax=b where b=inPsi, x=outPsi and A is DD^dagger
*/
void CG(SpinorField const& inPsi, SpinorField& outPsi); 

/*
    From multi-index (nt, nx, f, c) to flat index.
    The flattening is done such that c is the "innermost" index, nt is the "outhermost".
*/
unsigned int toFlat(int const nt, int const nx, int const f, int const c); 

/*
    Inverse operation of toFlat: converts a flattened index (innermost c, outermost nt)
    to a multi-index (nt, nx, f, c)
*/
std::vector<int> toVec(int const idx);

/*
    From multi-index (nt, nx, f, c) to flat even-odd index.
    The flattening is done such that c is the "innermost" index, nt is the "outhermost".
    Even comes first, odd after.
*/
unsigned int toEOflat(int const nt, int const nx, int const f, int const c);

/*
    Inverse operation of toEOflat: converts a flattened even-odd index (innermost c, outermost nt)
    to a multi-index (nt, nx, f, c)  
*/
std::vector<int> eoToVec(int const idx);

/*
    Perform matrix matrix product. M1 is on the left, M2 on the right. The result is stored into res
*/
void MatMatprod(mat const& M1, mat const& M2, mat& res);

/*
    Apply periodic boundary conditions to index with period N
*/
unsigned int PBC(int const n, int const N);
