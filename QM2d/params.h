#pragma once 

#include <complex.h>

int const Nt = 16;
int const Nx = 16;
int const Nf = 2;
int const vol = 2*Nf*Nx*Nt;

double const tol = 1e-12;
int const IterMax = 1000;

double const M = 0.5;

std::complex<double> const im (0.0, 1.0);