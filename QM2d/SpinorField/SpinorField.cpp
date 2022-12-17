#include "SpinorField.h"
#include <iostream>

SpinorField::SpinorField(int const Nt, int const Nx, int const Nf) :
    rndgen(rnddev()),
    dist(0., 1.),
    Nt{Nt},
    Nx{Nx},
    Nf{Nf},
    volume{2*Nf*Nt*Nx},
    val(2*Nt*Nx*Nf)
{
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            for(int f=0; f<Nf; f++){
                val[toFlat(nt, nx, f, 0)] = 1.3;
                val[toFlat(nt, nx, f, 1)] = 1.7;
            }
        }
    }
}

void SpinorField::DiracSpinorProduct(SpinorField& outPsi, bool dagger){
    assert(Nt == outPsi.Nt && Nx == outPsi.Nx && Nf == outPsi.Nf);
    std::vector<std::vector<std::complex<double>>> Gamma_p0 {{2, 0}, {0, 0}};
    std::vector<std::vector<std::complex<double>>> Gamma_m0 {{0, 0}, {0, 2}};
    std::vector<std::vector<std::complex<double>>> Gamma_p1 {{1, 1}, {1, 1}};
    std::vector<std::vector<std::complex<double>>> Gamma_m1 {{1, -1}, {-1, 1}};
    std::vector<std::vector<std::complex<double>>> const gamma5 {{0, im}, {-im, 0}};
    auto aux = gamma5;

    if (dagger){
        MatMatprod(Gamma_p0, gamma5, aux); MatMatprod(gamma5, aux, Gamma_p0);
        MatMatprod(Gamma_m0, gamma5, aux); MatMatprod(gamma5, aux, Gamma_m0);
        MatMatprod(Gamma_p1, gamma5, aux); MatMatprod(gamma5, aux, Gamma_p1);
        MatMatprod(Gamma_m1, gamma5, aux); MatMatprod(gamma5, aux, Gamma_m1);
    }

    std::cout << M << std::endl;

    double sgn[2];
    for(int nt=0; nt<Nt; nt++){
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        for(int nx=0; nx<Nx; nx++){
            for(int f=0; f<Nf; f++){
                outPsi.val[toFlat(nt, nx, f, 0)] = (2.0+M) * val[toFlat(nt, nx, f, 0)] 
                                                    - sgn[1]*0.5*Gamma_p0[0][0]*val[toFlat(PBC(nt-1, Nt), nx, f, 0)] - sgn[1]*0.5*Gamma_p0[0][1]*val[toFlat(PBC(nt-1, Nt), nx, f, 1)]
                                                    - sgn[0]*0.5*Gamma_m0[0][0]*val[toFlat(PBC(nt+1, Nt), nx, f, 0)] - sgn[0]*0.5*Gamma_m0[0][1]*val[toFlat(PBC(nt+1, Nt), nx, f, 1)]
                                                    - 0.5*Gamma_p1[0][0]*val[toFlat(nt, PBC(nx-1, Nx), f, 0)] - 0.5*Gamma_p1[0][1]*val[toFlat(nt, PBC(nx-1, Nx), f, 1)]
                                                    - 0.5*Gamma_m1[0][0]*val[toFlat(nt, PBC(nx+1, Nx), f, 0)] - 0.5*Gamma_m1[0][1]*val[toFlat(nt, PBC(nx+1, Nx), f, 1)];                                 
                outPsi.val[toFlat(nt, nx, f, 1)] = (2.0+M) * val[toFlat(nt, nx, f, 1)] 
                                                    - sgn[1]*0.5*Gamma_p0[1][0]*val[toFlat(PBC(nt-1, Nt), nx, f, 0)] - sgn[1]*0.5*Gamma_p0[1][1]*val[toFlat(PBC(nt-1, Nt), nx, f, 1)]
                                                    - sgn[0]*0.5*Gamma_m0[1][0]*val[toFlat(PBC(nt+1, Nt), nx, f, 0)] - sgn[0]*0.5*Gamma_m0[1][1]*val[toFlat(PBC(nt+1, Nt), nx, f, 1)]
                                                    - 0.5*Gamma_p1[1][0]*val[toFlat(nt, PBC(nx-1, Nx), f, 0)] - 0.5*Gamma_p1[1][1]*val[toFlat(nt, PBC(nx-1, Nx), f, 1)]
                                                    - 0.5*Gamma_m1[1][0]*val[toFlat(nt, PBC(nx+1, Nx), f, 0)] - 0.5*Gamma_m1[1][1]*val[toFlat(nt, PBC(nx+1, Nx), f, 1)];              
            }
        }
    }
}

std::complex<double> SpinorField::dot(SpinorField& s){
    std::complex<double> r = 0.0;
    for(int i=0; i<volume; i++) r += conj(val[i]) * s.val[i];
    return r;
}

