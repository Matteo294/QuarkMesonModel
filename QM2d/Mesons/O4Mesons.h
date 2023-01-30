#pragma once 

#include <vector>
#include <complex>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../functions/functions.h"
#include "../Lattice/Lattice.h"

using O4mat = Eigen::Matrix2cd;
class Lattice;

typedef std::vector<std::vector<O4mat>> O4field;

class O4Mesons {
    public:
        O4Mesons(double const m, double const lam, double const g, Lattice& l);
        O4Mesons(double const m, double const lam, double const g, double const sigma, double const pi[3], Lattice& l);
        ~O4Mesons(){;}
        double norm();
        O4field M;
        double const m2, lam, g;
        Lattice& l;
    
};