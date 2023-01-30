#include "DiracOP.h"

// maybe bring setZero() inside Dee Doo, ...

DiracOP::DiracOP(double const M, O4Mesons* mesons, Lattice& l) : 
    M{M}, 
    mesons{mesons},
    l{l}
    {;}



SpinorField DiracOP::applyTo(SpinorField const& inPsi, bool const dagger){
    assert(dagger==1 || dagger==0);
    
    SpinorField outPsi(inPsi.l);
    for(auto& v: outPsi.val) v.setZero();

    D_ee(inPsi, outPsi, dagger);
    D_oo(inPsi, outPsi, dagger);
    D_eo(inPsi, outPsi, dagger);
    D_oe(inPsi, outPsi, dagger);

    return outPsi;

}

SpinorField DiracOP::applyLDRTo(SpinorField const& inPsi, bool const dagger){
    assert(dagger==1 or dagger==0);

    SpinorField outPsi(inPsi.l);
    for(auto& v: outPsi.val) v.setZero();

    // !!!!!!!!!!!!!!!
    //Very inefficient function -> must be improved 
    // maybe pass iterators to even-odd parts of original vector instead of creating new full spinors of half volume
    // !!!!!!!!!!!!!!!!!!
 
    // Work out the even part
    SpinorField outEven(inPsi.l), aux1(inPsi.l), aux2(inPsi.l), aux3(inPsi.l);
    for(int i=0; i<inPsi.l.vol; i++) {outEven.val[i].setZero(); aux1.val[i].setZero(); aux2.val[i].setZero(); aux3.val[i].setZero();}

    D_ee(inPsi, outEven, dagger);
    D_oe(inPsi, aux1, dagger);
    D_oo_inv(aux1, aux2, dagger);
    D_eo(aux2, aux3, dagger);
    for(int i=0; i<inPsi.l.vol; i++) outEven.val[i] = outEven.val[i] - aux3.val[i];

    // Work out the odd part
    SpinorField outOdd(inPsi.l);
    for(auto& v: outOdd.val) v.setZero();
    D_oo(inPsi, outOdd, dagger);

    // Write to full spinor
    std::copy(outEven.val.begin(), outEven.val.begin() + inPsi.l.vol/2, outPsi.val.begin()); // write even part
    std::copy(outOdd.val.begin() + inPsi.l.vol/2, outOdd.val.end(), outPsi.val.begin() + inPsi.l.vol/2); // write odd part

    return outPsi;

}

SpinorField DiracOP::applyRto(SpinorField const& inPsi, bool const dagger){
    SpinorField outPsi(inPsi.l);
    for(auto& v: outPsi.val) v.setZero();

    SpinorField aux1(inPsi.l), aux2(inPsi.l);
    for(int i=0; i<inPsi.l.vol; i++) {aux1.val[i].setZero(); aux2.val[i].setZero();}

    D_oe(inPsi, aux1, dagger);
    D_oo_inv(aux1, aux2, dagger);

    std::copy(inPsi.val.begin(), inPsi.val.begin() + inPsi.l.vol/2, outPsi.val.begin()); // even part of the result
    for(int i=inPsi.l.vol/2; i<inPsi.l.vol; i++) outPsi.val[i] = inPsi.val[i] - aux2.val[i]; // odd part of the result

    return outPsi;
}

SpinorField DiracOP::applyLto(SpinorField const& inPsi, bool const dagger){
    SpinorField outPsi(inPsi.l);
    for(auto& v: outPsi.val) v.setZero();

    SpinorField aux1(inPsi.l), aux2(inPsi.l);
    for(int i=0; i<inPsi.l.vol; i++) {aux1.val[i].setZero(); aux2.val[i].setZero();}
    
    D_oo_inv(inPsi, aux1, dagger);
    D_eo(aux1, aux2, dagger);

    for(int i=0; i<inPsi.l.vol; i++) outPsi.val[i] = inPsi.val[i] - aux2.val[i]; // even part of the result
    std::copy(inPsi.val.begin() + inPsi.l.vol/2, inPsi.val.end(), outPsi.val.begin() + inPsi.l.vol/2); // odd part of the result

    return outPsi;
}

void DiracOP::D_oo_inv(SpinorField const& inPsi, SpinorField& outPsi, bool const dagger){
    std::complex<double> sigma;
    std::vector<int> idx(2);
    for(int i=inPsi.l.vol/2; i<inPsi.l.vol; i++){
        idx = inPsi.l.eoToVec(i);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace();
        outPsi.val[i].setZero();
        outPsi.val[i] += 1.0 / (2.0 + M + mesons->g*sigma) * inPsi.val[i];
    }
}


void DiracOP::D_ee(SpinorField const& inPsi, SpinorField& outPsi, bool const dagger){
    std::complex<double> sigma;
    std::vector<int> idx(2);

    // Diagonal term
    for(int i=0; i<inPsi.l.vol/2; i++){
        idx = inPsi.l.eoToVec(i);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace();
        outPsi.val[i] += (2.0 + M + mesons->g*sigma) * inPsi.val[i];
        if (dagger)
            outPsi.val[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()).adjoint(), gamma5) * inPsi.val[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            outPsi.val[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()), gamma5) * inPsi.val[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)

    }
}

void DiracOP::D_oo(SpinorField const& inPsi, SpinorField& outPsi, bool const dagger){
    std::complex<double> sigma;
    std::vector<int> idx(2);
    // Diagonal term
    for(int i=inPsi.l.vol/2; i<inPsi.l.vol; i++){
        idx = inPsi.l.eoToVec(i);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace();
        outPsi.val[i] += (2.0 + M + mesons->g*sigma) * inPsi.val[i];
        if (dagger)
            outPsi.val[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()).adjoint(), gamma5) * inPsi.val[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            outPsi.val[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()), gamma5) * inPsi.val[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)

    }
}
void DiracOP::D_eo(SpinorField const& inPsi, SpinorField& outPsi, bool const dagger){

    std::vector<int> idx(2);

    int const Nt=inPsi.l.Nt, Nx=inPsi.l.Nx;

    double sgn[2];
    int nt, nx;
    
    for(int i=0; i<inPsi.l.vol/2; i++){

        idx = inPsi.l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<double> psisum[2], psidiff[2];

        // maybe use ? : syntax to remove the if (more elegant?)
        
        if (dagger) {

            psisum[0] = inPsi.val[l.IDN[i][1]][0] + inPsi.val[l.IDN[i][1]][1];
            psisum[1] = inPsi.val[l.IDN[i][1]][2] + inPsi.val[l.IDN[i][1]][3];
            psidiff[0] = inPsi.val[l.IUP[i][1]][0] - inPsi.val[l.IUP[i][1]][1];
            psidiff[1] = inPsi.val[l.IUP[i][1]][2] - inPsi.val[l.IUP[i][1]][3];

            outPsi.val[l.toEOflat(nt, nx)][0] -=  sgn[1] * 1.0 * inPsi.val[l.IDN[i][0]][0] + 0.5*psidiff[0] + 0.5*psisum[0];
            outPsi.val[l.toEOflat(nt, nx)][2] -=  sgn[1] * 1.0 * inPsi.val[l.IDN[i][0]][2] + 0.5*psidiff[1] + 0.5*psisum[1];
            outPsi.val[l.toEOflat(nt, nx)][1] -=  sgn[0] * 1.0 * inPsi.val[l.IUP[i][0]][1] - 0.5*psidiff[0] + 0.5*psisum[0];
            outPsi.val[l.toEOflat(nt, nx)][3] -=  sgn[0] * 1.0 * inPsi.val[l.IUP[i][0]][3] - 0.5*psidiff[1] + 0.5*psisum[1];

        } else {

            psisum[0] = inPsi.val[l.IUP[i][1]][0] + inPsi.val[l.IUP[i][1]][1];
            psisum[1] = inPsi.val[l.IUP[i][1]][2] + inPsi.val[l.IUP[i][1]][3];
            psidiff[0] = inPsi.val[l.IDN[i][1]][0] - inPsi.val[l.IDN[i][1]][1];
            psidiff[1] = inPsi.val[l.IDN[i][1]][2] - inPsi.val[l.IDN[i][1]][3];

            outPsi.val[l.toEOflat(nt, nx)][0] -=  sgn[0] * 1.0 * inPsi.val[l.IUP[i][0]][0] + 0.5 * psisum[0] + 0.5 * psidiff[0];
            outPsi.val[l.toEOflat(nt, nx)][2] -=  sgn[0] * 1.0 * inPsi.val[l.IUP[i][0]][2] + 0.5 * psisum[1] + 0.5 * psidiff[1];
            outPsi.val[l.toEOflat(nt, nx)][1] -=  sgn[1] * 1.0 * inPsi.val[l.IDN[i][0]][1] + 0.5 * psisum[0] - 0.5 * psidiff[0];
            outPsi.val[l.toEOflat(nt, nx)][3] -=  sgn[1] * 1.0 * inPsi.val[l.IDN[i][0]][3] + 0.5 * psisum[1] - 0.5 * psidiff[1];

        }
                                            
    }

}

void DiracOP::D_oe(SpinorField const& inPsi, SpinorField& outPsi, bool const dagger){

    std::vector<int> idx(2);

    // Hopping term
    int const Nt=inPsi.l.Nt, Nx=inPsi.l.Nx;

    double sgn[2];
    int nt, nx;

    for(int i=inPsi.l.vol/2; i<inPsi.l.vol; i++){

        idx = inPsi.l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<double> psisum[2], psidiff[2];
        
        if (dagger) {

            psisum[0] = inPsi.val[l.IDN[i][1]][0] + inPsi.val[l.IDN[i][1]][1];
            psisum[1] = inPsi.val[l.IDN[i][1]][2] + inPsi.val[l.IDN[i][1]][3];
            psidiff[0] = inPsi.val[l.IUP[i][1]][0] - inPsi.val[l.IUP[i][1]][1];
            psidiff[1] = inPsi.val[l.IUP[i][1]][2] - inPsi.val[l.IUP[i][1]][3];

            outPsi.val[l.toEOflat(nt, nx)][0] -=  sgn[1] * 1.0 * inPsi.val[l.IDN[i][0]][0] + 0.5*psidiff[0] + 0.5*psisum[0];
            outPsi.val[l.toEOflat(nt, nx)][2] -=  sgn[1] * 1.0 * inPsi.val[l.IDN[i][0]][2] + 0.5*psidiff[1] + 0.5*psisum[1];
            outPsi.val[l.toEOflat(nt, nx)][1] -=  sgn[0] * 1.0 * inPsi.val[l.IUP[i][0]][1] - 0.5*psidiff[0] + 0.5*psisum[0];
            outPsi.val[l.toEOflat(nt, nx)][3] -=  sgn[0] * 1.0 * inPsi.val[l.IUP[i][0]][3] - 0.5*psidiff[1] + 0.5*psisum[1];

        } else {

            psisum[0] = inPsi.val[l.IUP[i][1]][0] + inPsi.val[l.IUP[i][1]][1];
            psisum[1] = inPsi.val[l.IUP[i][1]][2] + inPsi.val[l.IUP[i][1]][3];
            psidiff[0] = inPsi.val[l.IDN[i][1]][0] - inPsi.val[l.IDN[i][1]][1];
            psidiff[1] = inPsi.val[l.IDN[i][1]][2] - inPsi.val[l.IDN[i][1]][3];

            outPsi.val[l.toEOflat(nt, nx)][0] -=  sgn[0] * 1.0 * inPsi.val[l.IUP[i][0]][0] + 0.5 * psisum[0] + 0.5 * psidiff[0];
            outPsi.val[l.toEOflat(nt, nx)][2] -=  sgn[0] * 1.0 * inPsi.val[l.IUP[i][0]][2] + 0.5 * psisum[1] + 0.5 * psidiff[1];
            outPsi.val[l.toEOflat(nt, nx)][1] -=  sgn[1] * 1.0 * inPsi.val[l.IDN[i][0]][1] + 0.5 * psisum[0] - 0.5 * psidiff[0];
            outPsi.val[l.toEOflat(nt, nx)][3] -=  sgn[1] * 1.0 * inPsi.val[l.IDN[i][0]][3] + 0.5 * psisum[1] - 0.5 * psidiff[1];


        }
                                            
    }

}