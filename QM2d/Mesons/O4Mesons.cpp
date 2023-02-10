#include "O4Mesons.h"

O4Mesons::O4Mesons(double const m2, double const lam, double const g, Lattice& l) : 
    M(l.Nt*l.Nx, O4mat::Zero()),
    M_single(l.Nt*l.Nx, O4mat_single::Zero()),
    m2(m2),
    lam(lam),
    g{g},
    l{l}
    {;}

O4Mesons::O4Mesons(double const m, double const lam, double const g, double const sigma, double const pi[3], Lattice& l) :
    M(l.Nt*l.Nx),
    M_single(l.Nt*l.Nx),
    l{l},
    m2(m2),
    lam(lam),
    g{g}
{
    for(int i=0; i<l.vol; i++){  
        M[i] = O4mat {{sigma + im*pi[2], im*(pi[0] - im*pi[1])}, {im*(pi[0] + im*pi[1]), sigma - im*pi[2]}};
        M_single[i] = M[i].cast<std::complex<float>>();
    }
}

double O4Mesons::norm(){
    std::complex<double> det = 0.0;
    for(int i=0; i<l.vol; i++){
        det += sqrt(M[i].determinant());  
    }
    return det.real();
}

void O4Mesons::writeDoubleToSingle(){
    for(int i=0; i<l.vol; i++){
            M_single[i] = M[i].cast<std::complex<float>>();
        }
}

void O4Mesons::writeSingleToDouble(){
    for(int i=0; i<l.vol; i++){
            M[i] = M_single[i].cast<std::complex<double>>();
        }
}
