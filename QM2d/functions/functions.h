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

using mat = Eigen::Matrix2cd; // Matrix in spinor space OR flavour space (2x2)
using mat_fc = Eigen::Matrix<std::complex<double>, 2*Nf, 2*Nf>; // Matrix in spinor space tensor flavour space (composite matrix, 4x4)
using vec = Eigen::Vector2cd; // Vector in spinor space OR flavour space (2 components)
using vec_fc = Eigen::Vector<std::complex<double>, 2*Nf>; // Vector in spinor space tensor flavour space (4 components)

// Gamma5
mat const gamma5 {{0, im}, {-im, 0}};

/*
    Solve Ax=b where b=inPsi, x=outPsi and A is DD^dagger. One has to pass D, a Dirac operator.
    If hermitian is true -> use A=DdaggerD
*/
void CG(SpinorField const& inPsi, SpinorField& outPsi, DiracOP Dirac, bool hermitian=0); 

/*
    From multi-index (nt, nx, f, c) to flat even-odd index.
    The flattening is done such that c is the "innermost" index, nt is the "outhermost".
    Even comes first, odd after.
*/
unsigned int toEOflat(int const nt, int const nx, int const f, int const c);
unsigned int toEOflat(int const nt, int const nx);

/*
    Inverse operation of toEOflat: converts a flattened even-odd index (innermost c, outermost nt)
    to a multi-index (nt, nx, f, c)  
*/
std::vector<int> eoToVec(int const idx);

/*
    Apply periodic boundary conditions to index with period N
*/
unsigned int PBC(int const n, int const N);

/*
    Build a composite operator i.e. an operator in spinor tensor flavour space (4x4)
*/
mat_fc buildCompositeOP(mat const& flavourMat, mat const& spinorMat);
