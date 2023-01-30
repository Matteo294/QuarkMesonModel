#pragma once 

#include <complex>
#include <iostream>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


std::complex<double> constexpr im (0.0, 1.0);

using mat = Eigen::Matrix2cd; // Matrix in spinor space OR flavour space (2x2)
using mat_fc = Eigen::Matrix<std::complex<double>, 4, 4>; // Matrix in flavour tensor spinor space (composite matrix, 4x4)
using vec = Eigen::Vector2cd; // Vector in spinor space OR flavour space (2 components)
using vec_fc = Eigen::Vector<std::complex<double>, 4>; // Vector in flavour tensor spinor space (4 components)

//int IUP[Nt*Nx][2], IDN[Nt*Nx][2];

// Gamma5
mat const gamma5 {{0, im}, {-im, 0}};

/*
    Build a composite operator i.e. an operator in flavour tensor spinor space (4x4)
*/
mat_fc buildCompositeOP(mat const& flavourMat, mat const& spinorMat);