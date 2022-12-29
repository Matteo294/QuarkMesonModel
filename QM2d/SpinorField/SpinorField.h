#pragma once
#include <vector>
#include <complex>
#include <random>
#include <cassert> 
#include "../params.h"
#include "../functions/functions.h"
#include "../DiracOP/DiracOP.h"

typedef class SpinorField {
    public: 
        SpinorField(int const Nt, int const Nx, int const Nf);
        SpinorField(SpinorField const& s);
        ~SpinorField() {;}
        void operator = (SpinorField const&);
        std::vector<std::complex<double>> val;
        std::complex<double> dot(SpinorField& s); // dot current spinor with s (computes psi^dagger * s)
        int const Nt, Nx, Nf, volume;
    private:
        std::random_device rnddev;
        std::mt19937 rndgen;
        std::normal_distribution<double> dist;
} SpinorField;

