#pragma once 

#include <vector>
#include <complex>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../functions/functions.h"
#include "../Lattice/Lattice.h"

using O4mat = Eigen::Matrix2cd;
using O4mat_single = Eigen::Matrix2cf;

class Lattice;

typedef std::vector<O4mat> O4field;
typedef std::vector<O4mat_single> O4field_single;

class O4Mesons {
    public:
        O4Mesons(double const m, double const lam, double const g, Lattice& l);
        O4Mesons(double const m, double const lam, double const g, double const sigma, double const pi[3], Lattice& l);
        ~O4Mesons(){;}
        double norm();
        void writeDoubleToSingle(); // copy double into single
        void writeSingleToDouble(); // copy single into double
        O4field M;
        O4field_single M_single;
        double const m2, lam, g;
        Lattice& l;
    
};