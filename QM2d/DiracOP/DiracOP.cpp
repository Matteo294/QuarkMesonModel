#include "DiracOP.h"

DiracOP::DiracOP(double const M, O4Mesons* mesons) : 
    M{M}, 
    mesons{mesons},
    Gamma_p0 {{2, 0}, {0, 0}},
    Gamma_m0 {{0, 0}, {0, 2}},
    Gamma_p1 {{1, 1}, {1, 1}}, 
    Gamma_m1 {{1, -1}, {-1, 1}},
    Gamma_p0_dagger{gamma5*Gamma_p0*gamma5},
    Gamma_m0_dagger{gamma5*Gamma_m0*gamma5},
    Gamma_p1_dagger{gamma5*Gamma_p1*gamma5},
    Gamma_m1_dagger{gamma5*Gamma_m1*gamma5}
    {;}

SpinorField DiracOP::applyTo(SpinorField const& inPsi, bool dagger){
    
    SpinorField outPsi(inPsi.Nt, inPsi.Nx, inPsi.Nf);

    for(int i=0; i<outPsi.volume; i++) outPsi.val[i] = 0.0;

    D_ee(inPsi, outPsi); // dagger?
    D_oo(inPsi, outPsi); // dagger?
    D_eo(inPsi, outPsi, dagger);
    D_oe(inPsi, outPsi, dagger);

     
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){

            //for(int i=0; i<3; i++) std::cout << 0.5*(Pauli.tau[i]*mesons->M[nt][nx]).trace() << " ";
            //std::cout << "\n";

            // Store pions field values (including i factor)
            std::complex<double> pions[3];
            for(int i=0; i<3; i++) {
                pions[i] = 0.5*(Pauli.tau[i]*mesons->M[nt][nx]).trace();
            }

            // Prepare spinor at position (nt, nx)
            std::vector<std::complex<double>> x(4);
            for(int i=0; i<4; i++) x[i] = inPsi.val[toEOflat(nt, nx, 0, 0) + i];
            
            // Apply flavour mixing term to spinor
            mat F {{0, 0}, {0, 0}};
            for(int i=0; i<3; i++) F += mesons->g*pions[i]*Pauli.tau[i];
            //std::cout << F << "\n\n";
            auto v = applyFlavouredGamma5(F, x);
            if (dagger) v = applyFlavouredGamma5(F, x, 1);
            else v = applyFlavouredGamma5(F, x);

            // Sum to the result
            //for(int i=0; i<4; i++) std::cout << v[i] << " ";
            //std::cout << "\n";
            for(int i=0; i<4; i++) outPsi.val[toEOflat(nt, nx, 0, 0) + i] += v[i];
        }
    }

    return outPsi;

}

void DiracOP::D_ee(SpinorField const& inPsi, SpinorField& outPsi){
    for(int i=0; i<inPsi.volume/2; i++){
        auto idx = eoToVec(i);
        // Diagonal part (in flavour and spinor space)
        outPsi.val[i] += (Diag(idx[3],idx[3]) + 0.5*mesons->g*(Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace()) * inPsi.val[i];                                                            
    }
}
void DiracOP::D_oo(SpinorField const& inPsi, SpinorField& outPsi){
    for(int i=inPsi.volume/2; i<inPsi.volume; i++){
        auto idx = eoToVec(i);
        // Diagonal part (in flavour and spinor space)
        outPsi.val[i] += (Diag(idx[3],idx[3]) + 0.5*mesons->g*(Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace()) * inPsi.val[i];                                                            
    }
}
void DiracOP::D_eo(SpinorField const& inPsi, SpinorField& outPsi, bool dagger){
    auto aux = gamma5;
    mat _Gamma_m0{gamma5}, _Gamma_p0{gamma5}, _Gamma_m1{gamma5}, _Gamma_p1{gamma5};
    auto outPsi_copy = outPsi.val;

    // Note: to speed up, one could define the conjugate Gammas once for all
    if (dagger){
        _Gamma_p0 = gamma5*Gamma_p0*gamma5;
        _Gamma_m0 = gamma5*Gamma_m0*gamma5;
        _Gamma_p1 = gamma5*Gamma_p1*gamma5;
        _Gamma_m1 = gamma5*Gamma_m1*gamma5;
    } else {
        _Gamma_m0 = Gamma_m0; 
        _Gamma_p0 = Gamma_p0; 
        _Gamma_m1 = Gamma_m1; 
        _Gamma_p1 = Gamma_p1;
    }
    
    double sgn[2];
    for(int i=inPsi.volume/2; i<inPsi.volume; i++){
        auto idx = eoToVec(i);
        int nt=idx[0], nx=idx[1], f=idx[2], c=idx[3];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        for(int j=0; j<2; j++){
            outPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, j)] -= sgn[0]*0.5*_Gamma_p0(j,c)*inPsi.val[i];
            outPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, j)] -= sgn[1]*0.5*_Gamma_m0(j,c)*inPsi.val[i];
            outPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, j)] -= 0.5*_Gamma_p1(j,c)*inPsi.val[i];
            outPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, j)] -= 0.5*_Gamma_m1(j,c)*inPsi.val[i];
        }
    }
}

void DiracOP::D_oe(SpinorField const& inPsi, SpinorField& outPsi, bool dagger){
    auto aux = gamma5;
    mat _Gamma_m0{gamma5}, _Gamma_p0{gamma5}, _Gamma_m1{gamma5}, _Gamma_p1{gamma5};

    // Note: to speed up, one could define the conjugate Gammas once for all
   if (dagger){
        _Gamma_p0 = gamma5*Gamma_p0*gamma5;
        _Gamma_m0 = gamma5*Gamma_m0*gamma5;
        _Gamma_p1 = gamma5*Gamma_p1*gamma5;
        _Gamma_m1 = gamma5*Gamma_m1*gamma5;
    } else {
        _Gamma_m0 = Gamma_m0; 
        _Gamma_p0 = Gamma_p0; 
        _Gamma_m1 = Gamma_m1; 
        _Gamma_p1 = Gamma_p1;
    }
    
    double sgn[2];
    for(int i=inPsi.volume/2; i<inPsi.volume; i++){
        auto idx = eoToVec(i);
        int nt=idx[0], nx=idx[1], f=idx[2], c=idx[3];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        outPsi.val[i] +=    - sgn[1]*0.5*_Gamma_p0(c,0)*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 0)] - sgn[1]*0.5*_Gamma_p0(c,1)*inPsi.val[toEOflat(PBC(nt-1, Nt), nx, f, 1)]
                            - sgn[0]*0.5*_Gamma_m0(c,0)*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 0)] - sgn[0]*0.5*_Gamma_m0(c,1)*inPsi.val[toEOflat(PBC(nt+1, Nt), nx, f, 1)]
                            - 0.5*_Gamma_p1(c,0)*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 0)] - 0.5*_Gamma_p1(c,1)*inPsi.val[toEOflat(nt, PBC(nx-1, Nx), f, 1)]
                            - 0.5*_Gamma_m1(c,0)*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 0)] - 0.5*_Gamma_m1(c,1)*inPsi.val[toEOflat(nt, PBC(nx+1, Nx), f, 1)];
    }
    
}