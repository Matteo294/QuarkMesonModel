#pragma once 

#include "../params.h"
#include "../SpinorField/SpinorField.h"
#include "../DiracOP/DiracOP.h"

#include <complex>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

class SpinorField;
class DiracOP;

// Note: the following definitions IMPLICITLY ASSUME Nf=2 even though the rest of the code is written for general Nf
using mat = Eigen::Matrix2cd; // Matrix in spinor space OR flavour space (2x2)
using mat_fc = Eigen::Matrix4cd; // Matrix in spinor space tensor flavour space (composite matrix, 4x4)
using vec = Eigen::Vector2cd; // Vector in spinor space OR flavour space (2 components)
using vec_fc = Eigen::Vector4cd; // Vector in spinor space tensor flavour space (4 components)

// Gamma5
mat const gamma5 {{0, im}, {-im, 0}};

/*
    Solve Ax=b where b=inPsi, x=outPsi and A is DD^dagger. One has to pass D, a Dirac operator.
    If hermitian is true -> use A=DdaggerD
*/
void CG(SpinorField const& inPsi, SpinorField& outPsi, DiracOP Dirac, bool hermitian=0); 

/*
    From multi-index (nt, nx, f, c) to flat index.
    The flattening is done such that c is the "innermost" index, nt is the "outhermost".
*/
//unsigned int toFlat(int const nt, int const nx, int const f, int const c); 

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

/*
    Apply gamma5 in spinor space times a matrix (flavourMat) in flavour space
*/
vec_fc applyFlavouredGamma5(mat const flavourMat, std::vector<std::complex<double>> const& spinor);
