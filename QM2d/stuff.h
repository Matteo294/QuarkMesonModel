#pragma once 

#include <vector>
#include <complex>

int const Nt = 128, Nx = 16;

std::complex<double> const im {0, 1};

int const Nf = 2; // Number of flavours
int const vol = 2*Nx*Nt*Nf;

// CG params
int const IterMax = 1000;
double const tol = 1e-7;
double const M = 0.5;

void MatMatprod(std::vector<std::vector<std::complex<double>>> const& M1, std::vector<std::vector<std::complex<double>>> const& M2, std::vector<std::vector<std::complex<double>>>& res);
std::vector<int> toVec(int const idx);




