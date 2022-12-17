#pragma once
#include <vector>
#include <complex>
#include <random>
#include <cassert> 
#include "../stuff.h"

typedef class SpinorField {
    public: 
        int const Nt, Nx, Nf, volume;
        SpinorField(int const Nt, int const Nx, int const Nf);
        ~SpinorField() {;}
        std::vector<std::complex<double>> val;
        void DiracSpinorProduct(SpinorField& outPsi, bool dagger=0); // apply dirac operator to current spinor and stores it into outPsi
        std::complex<double> dot(SpinorField& s); // dot current spinor with s (computes psi^dagger * s
        unsigned int toFlat(int const nt, int const nx, int const f, int const c){return (c + 2*f + 2*nx*Nf + 2*nt*Nx*Nf);}


    private:
        std::random_device rnddev;
        std::mt19937 rndgen;
        std::normal_distribution<double> dist;
        unsigned int PBC(int const n, int const N){return (n+N)%N;}
} SpinorField;

