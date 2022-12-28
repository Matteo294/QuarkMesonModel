#include "SpinorField.h"
#include <iostream>

SpinorField::SpinorField(int const Nt, int const Nx, int const Nf) :
    rndgen(rnddev()),
    dist(0., 1.),
    Nt{Nt},
    Nx{Nx},
    Nf{Nf},
    volume{2*Nf*Nt*Nx},
    val(2*Nt*Nx*Nf, 0.0)
{
    for(int f=0; f<Nf; f++){
        val[toEOflat(0, 0, f, 0)] = 1.0;
        val[toEOflat(0, 0, f, 1)] = 1.0;
    }
}

SpinorField::SpinorField(SpinorField const& s) :
rndgen(rnddev()),
    dist(0., 1.),
    Nt{s.Nt},
    Nx{s.Nx},
    Nf{s.Nf},
    volume{s.volume},
    val(s.volume, 0.0)
{
    val = s.val;
}

void SpinorField::operator = (SpinorField const& s){
    assert(Nt == s.Nt && Nx == s.Nx && Nf == s.Nf);
    val = s.val;
}


std::complex<double> SpinorField::dot(SpinorField& s){
    std::complex<double> r = 0.0;
    for(int i=0; i<volume; i++) r += conj(val[i]) * s.val[i];
    return r;
}


SpinorField DiracSpinorProduct(SpinorField const& inPsi, bool dagger){
    
    //assert(Nt == outPsi.Nt && Nx == outPsi.Nx && Nf == outPsi.Nf);

    SpinorField outPsi(inPsi.Nt, inPsi.Nx, inPsi.Nf);

    std::vector<std::vector<std::complex<double>>> Gamma_p0 {{2, 0}, {0, 0}};
    std::vector<std::vector<std::complex<double>>> Gamma_m0 {{0, 0}, {0, 2}};
    std::vector<std::vector<std::complex<double>>> Gamma_p1 {{1, 1}, {1, 1}};
    std::vector<std::vector<std::complex<double>>> Gamma_m1 {{1, -1}, {-1, 1}};
    std::vector<std::vector<std::complex<double>>> const gamma5 {{0, im}, {-im, 0}};
    auto aux = gamma5;

    // Note: to speed, one could define the conjugate Gammas once for all
    if (dagger){
        MatMatprod(Gamma_p0, gamma5, aux); MatMatprod(gamma5, aux, Gamma_p0);
        MatMatprod(Gamma_m0, gamma5, aux); MatMatprod(gamma5, aux, Gamma_m0);
        MatMatprod(Gamma_p1, gamma5, aux); MatMatprod(gamma5, aux, Gamma_p1);
        MatMatprod(Gamma_m1, gamma5, aux); MatMatprod(gamma5, aux, Gamma_m1);
    }

    double sgn[2];
    for(int nt=0; nt<Nt; nt++){
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        for(int nx=0; nx<Nx; nx++){
            for(int f=0; f<Nf; f++){
                outPsi.val[toEOflat(nt, nx, f, 0)] = (2.0+M) * inPsi.val[toEOflat(nt, nx, f, 0)] 
                                                    - sgn[1]*0.5*Gamma_p0[0][0]*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 0)] - sgn[1]*0.5*Gamma_p0[0][1]*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 1)]
                                                    - sgn[0]*0.5*Gamma_m0[0][0]*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 0)] - sgn[0]*0.5*Gamma_m0[0][1]*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 1)]
                                                    - 0.5*Gamma_p1[0][0]*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 0)] - 0.5*Gamma_p1[0][1]*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 1)]
                                                    - 0.5*Gamma_m1[0][0]*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 0)] - 0.5*Gamma_m1[0][1]*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 1)];                                 
                outPsi.val[toEOflat(nt, nx, f, 1)] = (2.0+M) * inPsi.val[toEOflat(nt, nx, f, 1)] 
                                                    - sgn[1]*0.5*Gamma_p0[1][0]*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 0)] - sgn[1]*0.5*Gamma_p0[1][1]*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 1)]
                                                    - sgn[0]*0.5*Gamma_m0[1][0]*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 0)] - sgn[0]*0.5*Gamma_m0[1][1]*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 1)]
                                                    - 0.5*Gamma_p1[1][0]*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 0)] - 0.5*Gamma_p1[1][1]*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 1)]
                                                    - 0.5*Gamma_m1[1][0]*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 0)] - 0.5*Gamma_m1[1][1]*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 1)];              
            }
        }
    }

    return outPsi;

}


