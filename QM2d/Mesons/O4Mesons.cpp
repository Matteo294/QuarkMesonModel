#include "O4Mesons.h"

O4Mesons::O4Mesons(int const Nt, int const Nx, double const m2, double const lam) : 
    phi(Nt, std::vector<Eigen::Vector4cd> (Nx)),
    Nt{Nt},
    Nx{Nx},
    m2(m2),
    lam(lam)
{
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            phi[nt][nx] = Eigen::Vector4cd {1.0*nt, 2.0, 3.0, 4.0};
        }
    }
}

std::complex<double> O4Mesons::evaluateDrift(int const nt, int const nx, int const j){
    std::complex<double> K = - phi[PBC(nt+1, Nt)][nx][j] - phi[PBC(nt-1, Nt)][nx][j] - phi[nt][PBC(nx+1, Nx)][j] - phi[nt][PBC(nx-1, Nx)][j] + phi[nt][nx][j] * (m2 + 2) - lam/6.0 * pow(phi[nt][nx][j], 3);
    for(int i=0; i<4; i++){
        if (i != j){
            K -= lam/6.0 * phi[nt][nx][j] * pow(phi[nt][nx][i], 2);
        }
    }
    return K;
}

double O4Mesons::norm(){
    double n = 0.0;
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            n += sqrt((phi[nt][nx][0]*conj(phi[nt][nx][0]) + phi[nt][nx][1]*conj(phi[nt][nx][1]) + phi[nt][nx][2]*conj(phi[nt][nx][2]) + phi[nt][nx][3]*conj(phi[nt][nx][3])).real());
        }
    }
    return n/(Nt*Nx);
}
