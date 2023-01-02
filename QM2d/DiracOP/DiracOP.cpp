#include "DiracOP.h"

DiracOP::DiracOP(double const M, O4Mesons* mesons) : M{M}, mesons{mesons} {;}

SpinorField DiracOP::applyTo(SpinorField const& inPsi, bool dagger){
    
    SpinorField outPsi(inPsi.Nt, inPsi.Nx, inPsi.Nf);
    auto aux = gamma5;
    mat _Gamma_m0=gamma5, _Gamma_p0=gamma5, _Gamma_m1=gamma5, _Gamma_p1=gamma5;

    // Note: to speed up, one could define the conjugate Gammas once for all
    if (dagger){
        MatMatprod(Gamma_p0, gamma5, aux); MatMatprod(gamma5, aux, _Gamma_p0);
        MatMatprod(Gamma_m0, gamma5, aux); MatMatprod(gamma5, aux, _Gamma_m0);
        MatMatprod(Gamma_p1, gamma5, aux); MatMatprod(gamma5, aux, _Gamma_p1);
        MatMatprod(Gamma_m1, gamma5, aux); MatMatprod(gamma5, aux, _Gamma_m1);
    } else {
        _Gamma_m0 = Gamma_m0; 
        _Gamma_p0 = Gamma_p0; 
        _Gamma_m1 = Gamma_m1; 
        _Gamma_p1 = Gamma_p1;
    }

    double sgn[2];
    for(int nt=0; nt<Nt; nt++){
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        for(int nx=0; nx<Nx; nx++){

            // Free fermions part
            for(int f=0; f<Nf; f++){
                outPsi.val[toEOflat(nt, nx, f, 0)] = Diag[0][0] * inPsi.val[toEOflat(nt, nx, f, 0)] 
                                                    - sgn[1]*0.5*_Gamma_p0[0][0]*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 0)] - sgn[1]*0.5*_Gamma_p0[0][1]*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 1)]
                                                    - sgn[0]*0.5*_Gamma_m0[0][0]*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 0)] - sgn[0]*0.5*_Gamma_m0[0][1]*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 1)]
                                                    - 0.5*_Gamma_p1[0][0]*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 0)] - 0.5*_Gamma_p1[0][1]*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 1)]
                                                    - 0.5*_Gamma_m1[0][0]*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 0)] - 0.5*_Gamma_m1[0][1]*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 1)];                                 
                outPsi.val[toEOflat(nt, nx, f, 1)] = Diag[1][1] * inPsi.val[toEOflat(nt, nx, f, 1)] 
                                                    - sgn[1]*0.5*_Gamma_p0[1][0]*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 0)] - sgn[1]*0.5*_Gamma_p0[1][1]*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 1)]
                                                    - sgn[0]*0.5*_Gamma_m0[1][0]*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 0)] - sgn[0]*0.5*_Gamma_m0[1][1]*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 1)]
                                                    - 0.5*_Gamma_p1[1][0]*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 0)] - 0.5*_Gamma_p1[1][1]*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 1)]
                                                    - 0.5*_Gamma_m1[1][0]*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 0)] - 0.5*_Gamma_m1[1][1]*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 1)];              
            }

            // Yukawa interaction with mesons
            outPsi.val[toEOflat(nt, nx, 0, 0)] += mesons->phi[nt][nx][0]    * inPsi.val[toEOflat(nt, nx, 0, 0)] 
                                                + im*mesons->phi[nt][nx][1] * (gamma5[0][0]*inPsi.val[toEOflat(nt, nx, 1, 0)] + gamma5[0][1]*inPsi.val[toEOflat(nt, nx, 1, 1)])
                                                +    mesons->phi[nt][nx][2] * (gamma5[0][0]*inPsi.val[toEOflat(nt, nx, 1, 0)] + gamma5[0][1]*inPsi.val[toEOflat(nt, nx, 1, 1)])
                                                + im*mesons->phi[nt][nx][3] * (gamma5[0][0]*inPsi.val[toEOflat(nt, nx, 0, 0)] + gamma5[0][1]*inPsi.val[toEOflat(nt, nx, 0, 1)]);
            outPsi.val[toEOflat(nt, nx, 0, 1)] += mesons->phi[nt][nx][0]    * inPsi.val[toEOflat(nt, nx, 0, 1)] 
                                                + im*mesons->phi[nt][nx][1] * (gamma5[1][0]*inPsi.val[toEOflat(nt, nx, 1, 0)] + gamma5[1][1]*inPsi.val[toEOflat(nt, nx, 1, 1)])
                                                +    mesons->phi[nt][nx][2] * (gamma5[1][0]*inPsi.val[toEOflat(nt, nx, 1, 0)] + gamma5[1][1]*inPsi.val[toEOflat(nt, nx, 1, 1)])
                                                + im*mesons->phi[nt][nx][3] * (gamma5[1][0]*inPsi.val[toEOflat(nt, nx, 0, 0)] + gamma5[1][1]*inPsi.val[toEOflat(nt, nx, 0, 1)]);
            outPsi.val[toEOflat(nt, nx, 1, 0)] += mesons->phi[nt][nx][0]    * inPsi.val[toEOflat(nt, nx, 1, 0)] 
                                                + im*mesons->phi[nt][nx][1] * (gamma5[0][0]*inPsi.val[toEOflat(nt, nx, 0, 0)] + gamma5[0][1]*inPsi.val[toEOflat(nt, nx, 0, 1)])
                                                -    mesons->phi[nt][nx][2] * (gamma5[0][0]*inPsi.val[toEOflat(nt, nx, 0, 0)] + gamma5[0][1]*inPsi.val[toEOflat(nt, nx, 0, 1)])
                                                - im*mesons->phi[nt][nx][3] * (gamma5[0][0]*inPsi.val[toEOflat(nt, nx, 1, 0)] + gamma5[0][1]*inPsi.val[toEOflat(nt, nx, 1, 1)]);
            outPsi.val[toEOflat(nt, nx, 1, 1)] += mesons->phi[nt][nx][0]    * inPsi.val[toEOflat(nt, nx, 0, 1)] 
                                                + im*mesons->phi[nt][nx][1] * (gamma5[1][0]*inPsi.val[toEOflat(nt, nx, 0, 0)] + gamma5[1][1]*inPsi.val[toEOflat(nt, nx, 0, 1)])
                                                -    mesons->phi[nt][nx][2] * (gamma5[1][0]*inPsi.val[toEOflat(nt, nx, 0, 0)] + gamma5[1][1]*inPsi.val[toEOflat(nt, nx, 0, 1)])
                                                - im*mesons->phi[nt][nx][3] * (gamma5[1][0]*inPsi.val[toEOflat(nt, nx, 1, 0)] + gamma5[1][1]*inPsi.val[toEOflat(nt, nx, 1, 1)]);

        }
    }

    return outPsi;

}