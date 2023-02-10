#pragma once 

// Lattice params
int constexpr Nt = 128;
int constexpr Nx = 32;
int constexpr Nf = 2; // works only for Nf = 2
int constexpr vol = 2*Nf*Nx*Nt;

// CG params
double constexpr tol = 1e-12;
int constexpr IterMax = 2000;
int constexpr CGmode = 0; // 0: double, 1: EO double, 2: single, 3: EO single, 4: mixed, 5: EO mixed
int constexpr IterMaxSingle = 100; // Useful only in mixed precision CG (see notes)

// Action params
double constexpr fermion_M = 0.1; // fermions mass
double constexpr meson_M2 = -1.0; // scalars mass (both sigma and pions)
double constexpr lam = 1.0; // quartic coupling
double constexpr g = 3.4; // Yukawa coupling

// Langevin params
double constexpr dt = 0.01;
double constexpr T = 1.0;

// Initial field values
double constexpr sigma = 0.01;
double constexpr pi[3] {0.02, 0.03, 0.04};