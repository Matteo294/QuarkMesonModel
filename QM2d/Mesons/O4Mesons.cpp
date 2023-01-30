#include "O4Mesons.h"

O4Mesons::O4Mesons(double const m2, double const lam, double const g, Lattice& l) : 
    M(l.Nt, std::vector<O4mat> (l.Nx)),
    m2(m2),
    lam(lam),
    g{g},
    l{l}
{
    for(int nt=0; nt<l.Nt; nt++){
        for(int nx=0; nx<l.Nx; nx++){
            M[nt][nx] = O4mat {{0, 0}, {0, 0}};
        }
    }
}

O4Mesons::O4Mesons(double const m, double const lam, double const g, double const sigma, double const pi[3], Lattice& l) :
    M(l.Nt, std::vector<O4mat> (l.Nx)),
    l{l},
    m2(m2),
    lam(lam),
    g{g}
{
    for(int nt=0; nt<l.Nt; nt++){
        for(int nx=0; nx<l.Nx; nx++){
            M[nt][nx] = O4mat {{sigma + im*pi[2], im*(pi[0] - im*pi[1])}, {im*(pi[0] + im*pi[1]), sigma - im*pi[2]}};
        }
    }
}

double O4Mesons::norm(){
    std::complex<double> det = 0.0;
    for(int nt=0; nt<l.Nt; nt++){
        for(int nx=0; nx<l.Nx; nx++){
            det += sqrt(M[nt][nx].determinant());
        }
    }
    return det.real();
}
