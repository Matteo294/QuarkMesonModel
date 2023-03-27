#pragma once 

#include <vector>
#include <complex>
#include "../functions/functions.h"
#include "../Lattice/Lattice.h"

template <typename Tvar>
class O4Mesons {
    public:
        O4Mesons(double const m, double const lam, double const g, Lattice& l);
        O4Mesons(double const m, double const lam, double const g, std::complex<double> const sigma, std::complex<double> const pi[3], Lattice& l);
        ~O4Mesons(){;}
        double norm();
        std::vector<MesonsMat<Tvar>> M;
        double const g;
    private:
        double const m2, lam;
        int const vol;
    
};

typedef O4Mesons<double> O4Mesons_d;
typedef O4Mesons<float> O4Mesons_f;

template <typename Tvar>
O4Mesons<Tvar>::O4Mesons(double const m2, double const lam, double const g, Lattice& l) : 
    M(l.vol),
    m2(m2),
    lam(lam),
    g{g},
    vol{l.vol}
    {;}

template <typename Tvar>
O4Mesons<Tvar>::O4Mesons(double const m, double const lam, double const g, std::complex<double> const sigma, std::complex<double> const pi[3], Lattice& l) :
    M(l.vol),
    m2(m2),
    lam(lam),
    vol{l.vol},
    g{g}
{
    for(int i=0; i<l.vol; i++){  
        std::complex<Tvar> sigmacopy = (std::complex<Tvar>) sigma;
        std::complex<Tvar> picopy[3]; for(int i=0; i<3; i++) picopy[i] =  (std::complex<Tvar>) pi[i];
        M[i] = MesonsMat<Tvar>(sigmacopy, picopy);
        std::complex<float> buf[3] {(std::complex<float>) pi[0], (std::complex<float>) pi[1], (std::complex<float>) pi[2]};
    }
}

template <typename Tvar>
double O4Mesons<Tvar>::norm(){
    std::complex<double> det = 0.0;
    for(int i=0; i<vol; i++){
        det += sqrt(M[i].val[0][0]*M[i].val[1][1] - M[i].val[1][0]*M[i].val[0][1]);  // not sure whether sqrt is correct here; probably shoud be outside sum
    }
    return det.real();
}
