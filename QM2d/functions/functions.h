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

enum class MatrixType { Normal, Dagger };
using spinor_iter = std::vector<vec_fc>::iterator;
using spinor_single_iter = std::vector<vec_fc_single>::iterator;

// Gamma5
mat const gamma5 {{0, im}, {-im, 0}};

std::complex<double> dotProduct(spinor_iter v1begin, spinor_iter v1end, spinor_iter v2begin);
std::complex<float> dotProduct(spinor_single_iter v1begin, spinor_single_iter v1end, spinor_single_iter v2begin);

typedef void (*ptrMat) (spinor_iter vec, spinor_iter res, MatrixType const useDagger);