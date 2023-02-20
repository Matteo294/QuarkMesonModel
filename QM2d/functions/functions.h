#pragma once 

#include <complex>
#include <iostream>
#include <vector>

struct Spinor {
    std::vector<std::complex<double>> val {0.0, 0.0, 0.0, 0.0};
};

struct O4Mat {
    O4Mat();
    O4Mat(std::complex<double> const sigma, std::complex<double> const pi[3]);
    std::complex<double> determinant();
    std::vector<std::vector<std::complex<double>>> val;
};

struct O4Mat_single {
    O4Mat_single();
    O4Mat_single(std::complex<float> sigma, std::complex<float> pi[3]);
    std::vector<std::vector<std::complex<float>>> val;
};


std::complex<double> constexpr im (0.0, 1.0);
std::complex<float> constexpr imf (0.0, 1.0);

enum class MatrixType { Normal, Dagger };

using vecfield = std::vector<Spinor>;
using vecfield_iter = vecfield::iterator;
using vecfield_single = std::vector<std::vector<std::complex<float>>>;
using vecfield_single_iter = vecfield_single::iterator;

using mat_single = std::vector<std::vector<std::complex<float>>>;


std::complex<double> dotProduct(vecfield_iter v1begin, vecfield_iter v1end, vecfield_iter v2begin);
std::complex<float> dotProduct(vecfield_single_iter v1begin, vecfield_single_iter v1end, vecfield_single_iter v2begin);
std::complex<double> dotProduct(std::vector<std::complex<double>>::iterator v1begin, std::vector<std::complex<double>>::iterator v1end, std::vector<std::complex<double>>::iterator v2begin);

void spinorSum(vecfield_iter s1begin, vecfield_iter s1end, vecfield_iter s2begin, vecfield_iter resbegin);
void spinorDiff(vecfield_iter s1begin, vecfield_iter s1end, vecfield_iter s2begin, vecfield_iter resbegin);

typedef void (*ptrMat) (vecfield_iter vec, vecfield_iter res, MatrixType const useDagger);

