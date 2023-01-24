#pragma once
#include <vector>
#include <complex>
#include <random>
#include <cassert> 
#include "../params.h"
#include "../functions/functions.h"
#include "../DiracOP/DiracOP.h"

using mat = Eigen::Matrix2cd; // Matrix in spinor space OR flavour space (2x2)
using mat_fc = Eigen::Matrix<std::complex<double>, 2*Nf, 2*Nf>; // Matrix in spinor space tensor flavour space (composite matrix, 4x4)
using vec = Eigen::Vector2cd; // Vector in spinor space OR flavour space (2 components)
using vec_fc = Eigen::Vector<std::complex<double>, 2*Nf>; // Vector in spinor space tensor flavour space (4 components)

typedef class SpinorField {
    public: 
        SpinorField(int const Nt, int const Nx, int const Nf);
        SpinorField(SpinorField const& s);
        ~SpinorField() {;}
        void operator = (SpinorField const&);
        std::vector<vec_fc> val;
        std::complex<double> dot(SpinorField& s); // dot current spinor with s (computes psi^dagger * s)
        int const Nt, Nx, Nf, volume;
    private:
        std::random_device rnddev;
        std::mt19937 rndgen;
        std::normal_distribution<double> dist;
} SpinorField;

