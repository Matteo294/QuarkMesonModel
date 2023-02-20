#pragma once 

#include <vector>
#include <complex>
#include "../functions/functions.h"
#include "../Lattice/Lattice.h"

struct Spinor;

class Lattice;



typedef std::vector<O4Mat> O4field;
typedef std::vector<O4Mat_single> O4field_single;

class O4Mesons {
    public:
        O4Mesons(double const m, double const lam, double const g, Lattice& l);
        O4Mesons(double const m, double const lam, double const g, std::complex<double> const sigma, std::complex<double> const pi[3], Lattice& l);
        ~O4Mesons(){;}
        double norm();
        O4field M;
        O4field_single M_single;
    
    double const m2, lam, g;
    int const vol;
    
};