#pragma once 

#include <vector>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "../functions/functions.h"

// Not flattened. No flavour.

typedef std::vector<std::vector<Eigen::Vector4cd>> O4field;

class O4Mesons {
    public:
        O4Mesons(int const Nt, int const Nx, double const m, double const lam);
        ~O4Mesons(){;}
        std::complex<double> evaluateDrift(int const nt, int const nx, int const j);
        double norm();
        O4field phi;
        int const Nt, Nx;
        double const m2, lam;
    
};