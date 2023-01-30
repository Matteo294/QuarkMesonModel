#pragma once 

#include <complex.h>

// Lattice params
int constexpr Nt = 128;
int constexpr Nx = 32;
int constexpr Nf = 2; // works only for Nf = 2
int constexpr vol = 2*Nf*Nx*Nt;

// CG params
double constexpr tol = 1e-12;
int constexpr IterMax = 2000;

// Action params
double constexpr fermion_M = 0.1; // fermions mass
double constexpr meson_M2 = -1.0; // scalars mass (both sigma and pions)
double constexpr lam = 1.0; // quartic coupling
double constexpr g = 3.4; // Yukawa coupling

// Langevin params
double constexpr dt = 0.01;
double constexpr T = 1.0;

// Others
//std::complex<double> constexpr im (0.0, 1.0); 

// Initial field values
double constexpr sigma = 0.01;
//double constexpr sigma = 0.0;
//double constexpr pi[3] {0.02, 0.03, 0.04};
double constexpr pi[3] {0, 0, 0.0};