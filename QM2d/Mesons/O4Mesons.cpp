#include "O4Mesons.h"
#include <algorithm>

O4Mesons::O4Mesons(double const m2, double const lam, double const g, Lattice& l) : 
    M(l.vol),
    M_single(l.vol),
    m2(m2),
    lam(lam),
    g{g},
    vol{l.vol}
    {;}

O4Mesons::O4Mesons(double const m, double const lam, double const g, std::complex<double> const sigma, std::complex<double> const pi[3], Lattice& l) :
    M(l.vol),
    M_single(l.vol),
    m2(m2),
    lam(lam),
    vol{l.vol},
    g{g}
{
    for(int i=0; i<l.vol; i++){  
        M[i] = O4Mat(sigma, pi);
        std::complex<float> buf[3] {(std::complex<float>) pi[0], (std::complex<float>) pi[1], (std::complex<float>) pi[2]};
        M_single[i] = O4Mat_f((std::complex<float>) sigma, buf);
    }
}

double O4Mesons::norm(){
    std::complex<double> det = 0.0;
    for(int i=0; i<vol; i++){
        det += sqrt(M[i].val[0][0]*M[i].val[1][1] - M[i].val[1][0]*M[i].val[0][1]);  // not sure whether sqrt is correct here; probably shoud be outside sum
    }
    return det.real();
}

