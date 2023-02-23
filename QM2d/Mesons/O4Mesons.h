#pragma once 

#include <vector>
#include <complex>
#include "../functions/functions.h"
#include "../Lattice/Lattice.h"

using O4field = std::vector<O4Mat<double>>;
using O4field_single = std::vector<O4Mat<float>>;

typedef class O4Mesons {
    public:
        O4Mesons(double const m, double const lam, double const g, Lattice& l);
        O4Mesons(double const m, double const lam, double const g, std::complex<double> const sigma, std::complex<double> const pi[3], Lattice& l);
        ~O4Mesons(){;}
        double norm();
        O4field M;
        O4field_single M_single;
    
    double const m2, lam, g;
    int const vol;
    
} O4Mesons;