#pragma once
#include <vector>
#include <complex>
#include <random>
#include <cassert> 
#include "../functions/functions.h"
#include "../DiracOP/DiracOP.h"
#include "../Lattice/Lattice.h"

using mat = Eigen::Matrix2cd; // Matrix in spinor space OR flavour space (2x2)
using mat_fc = Eigen::Matrix<std::complex<double>, 4, 4>; // Matrix in spinor space tensor flavour space (composite matrix, 4x4)
using vec = Eigen::Vector2cd; // Vector in spinor space OR flavour space (2 components)
using vec_fc = Eigen::Vector<std::complex<double>, 4>; // Vector in spinor space tensor flavour space (4 components)

class Lattice;

typedef class SpinorField {
    public: 
        SpinorField(Lattice& l);
        SpinorField(SpinorField const& s);
        ~SpinorField() {;}
        void operator = (SpinorField const&);
        std::vector<vec_fc> val;
        std::complex<double> dot(SpinorField& s); // dot current spinor with s (computes psi^dagger * s)
        Lattice& l;
    private:
        std::random_device rnddev;
        std::mt19937 rndgen;
        std::normal_distribution<double> dist;
} SpinorField;

