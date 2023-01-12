#pragma once 

#include <vector>
#include <complex>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../functions/functions.h"

// Not flattened. No flavour.

using O4mat = Eigen::Matrix2cd;

typedef std::vector<std::vector<O4mat>> O4field;

class O4Mesons {
    public:
        O4Mesons(int const Nt, int const Nx, double const m, double const lam, double const g);
        O4Mesons(int const Nt, int const Nx, double const m, double const lam, double const g, double const sigma, double const pi[3]);
        ~O4Mesons(){;}
        O4mat evaluateDrift(int const nt, int const nx);
        double norm();
        O4field M;
        int const Nt, Nx;
        double const m2, lam, g;
    
};