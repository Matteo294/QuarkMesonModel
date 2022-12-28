#pragma once
#include <vector>
#include <complex>
#include <random>
#include <cassert> 
#include "../params.h"
#include "../functions/functions.h"


typedef class SpinorField {
    public: 
        int const Nt, Nx, Nf, volume;
        SpinorField(int const Nt, int const Nx, int const Nf);
        SpinorField(SpinorField const& s);
        void operator = (SpinorField const&);
        ~SpinorField() {;}
        std::vector<std::complex<double>> val;
        std::complex<double> dot(SpinorField& s); // dot current spinor with s (computes psi^dagger * s)
    private:
        std::random_device rnddev;
        std::mt19937 rndgen;
        std::normal_distribution<double> dist;
} SpinorField;

SpinorField DiracSpinorProduct(SpinorField const& inPsi, bool dagger=0);
