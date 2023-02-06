#pragma once 

#include <complex>
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


std::complex<double> constexpr im (0.0, 1.0);

using mat = Eigen::Matrix2cd; // Matrix in spinor space OR flavour space (2x2)
using mat_fc = Eigen::Matrix<std::complex<double>, 4, 4>; // Matrix in flavour tensor spinor space (composite matrix, 4x4)
using vec = Eigen::Vector2cd; // Vector in spinor space OR flavour space (2 components)
using vec_fc = Eigen::Vector<std::complex<double>, 4>; // Vector in flavour tensor spinor space (4 components)
using mat_fc_single = Eigen::Matrix<std::complex<float>, 4, 4>;
using vec_fc_single = Eigen::Vector<std::complex<float>, 4>;
using mat_single = Eigen::Matrix2cf; // Matrix in spinor space OR flavour space (2x2)
using vec_single = Eigen::Vector2cf; // Vector in spinor space OR flavour space (2 components)



// Gamma5
mat const gamma5 {{0, im}, {-im, 0}};

/*
    Build a composite operator i.e. an operator in flavour tensor spinor space (4x4)
*/
mat_fc buildCompositeOP(mat const& flavourMat, mat const& spinorMat);
mat_fc_single buildCompositeOP_single(mat_single const& flavourMat, mat_single const& spinorMat);


std::complex<double> dotProduct(std::vector<vec_fc>::iterator v1begin, std::vector<vec_fc>::iterator v1end, std::vector<vec_fc>::iterator v2begin);
std::complex<float> dotProduct(std::vector<vec_fc_single>::iterator v1begin, std::vector<vec_fc_single>::iterator v1end, std::vector<vec_fc_single>::iterator v2begin);