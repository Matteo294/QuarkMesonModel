#pragma once 

#include <complex.h>

// Lattice params
int constexpr Nt = 128;
int const Nx = 16;
int const Nf = 2; // works only for Nf = 2
int const vol = 2*Nf*Nx*Nt;

// CG params
double const tol = 1e-12;
int const IterMax = 1000;

// Action params
double const fermion_M = 0.03; // fermions mass
double const meson_M2 = -1.0; // scalars mass (both sigma and pions)
double const lam = 1.0; // quartic coupling
double const g = 3.4; // Yukawa coupling

// Langevin params
double const dt = 0.01;
double const T = 1.0;

// Others
std::complex<double> const im (0.0, 1.0);
int const Ntherm = 100;
int const Ndata = 100;  

// Initial field values
double const sigma = 0.01;
//double const sigma = 0.0;
double const pi[3] {0.02, 0.03, 0.04};
//double const pi[3] {0, 0, 0.0};