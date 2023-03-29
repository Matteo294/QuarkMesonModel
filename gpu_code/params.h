#pragma once

int constexpr Nt = 6, Nx = 60;

enum class MatrixType {Normal, Dagger};

double constexpr fermion_mass = 0.05;
double constexpr g_coupling = 0.0;
double constexpr sigma = 0.2;
double constexpr pi[] = {0.1, 0.2, 0.3};

double constexpr tolerance = 1e-12;
int constexpr IterMax = 1000;
