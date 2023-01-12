#include "O4Mesons.h"

O4Mesons::O4Mesons(int const Nt, int const Nx, double const m2, double const lam, double const g) : 
    M(Nt, std::vector<O4mat> (Nx)),
    Nt{Nt},
    Nx{Nx},
    m2(m2),
    lam(lam),
    g{g}
{
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            M[nt][nx] = O4mat {{0, 0}, {0, 0}};
        }
    }
}

O4Mesons::O4Mesons(int const Nt, int const Nx, double const m, double const lam, double const g, double const sigma, double const pi[3]) :
    M(Nt, std::vector<O4mat> (Nx)),
    Nt{Nt},
    Nx{Nx},
    m2(m2),
    lam(lam),
    g{g}
{
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            M[nt][nx] = O4mat {{sigma + im*pi[2], im*(pi[0] - im*pi[1])}, {im*(pi[0] + im*pi[1]), sigma - im*pi[2]}};
        }
    }
}


O4mat O4Mesons::evaluateDrift(int const nt, int const nx){
    return M[PBC(nt+1, Nt)][nx] + M[PBC(nt-1, Nt)][nx] + M[nt][PBC(nx+1, Nx)] + M[nt][PBC(nx-1, Nx)] - (4+m2)*M[nt][nx] - lam/6.0*M[nt][nx].determinant()*M[nt][nx];
}

double O4Mesons::norm(){
    std::complex<double> det = 0.0;
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            det += sqrt(M[nt][nx].determinant());
        }
    }
    return det.real();
}
