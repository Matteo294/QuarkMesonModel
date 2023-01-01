#pragma once 

#include <complex.h>

// Lattice params
int const Nt = 16;
int const Nx = 16;
int const Nf = 2;
int const vol = 2*Nf*Nx*Nt;

// CG params
double const tol = 1e-12;
int const IterMax = 1000;

// Fermions params
double const fermion_M = 0.5;

// Mesons params
double const meson_M2 = -1.0;
double const lam = 1.0;

// Langevin params
double const dt = 0.01;
double const T = 1.0;

// Others
std::complex<double> const im (0.0, 1.0);