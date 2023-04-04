#pragma once

int constexpr Nt = 64, Nx = 64;

enum class MatrixType {Normal, Dagger};

double constexpr fermion_mass = 0.5;
double constexpr g_coupling = 0.1;
double constexpr sigma = 0.0;
double constexpr pi[] = {0.0, 0.0, 0.3};

double constexpr tolerance = 1e-12;
int constexpr IterMax = 1000;
