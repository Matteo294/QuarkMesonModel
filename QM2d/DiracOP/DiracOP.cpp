#include "DiracOP.h"

// maybe bring setZero() inside Dee Doo, ...

DiracOP::DiracOP(double const M, O4Mesons& mesons, Lattice& l) : 
    M{M}, 
    mesons{mesons},
    l{l}
    {;}

template <typename T>
void DiracOP::applyTo(T vec, T res, MatrixType const useDagger){
    for(int i=0; i<l.vol; i++) {res[i].setZero();}
    D_ee(vec, res, useDagger);
    D_oo(vec + l.vol/2, res + l.vol/2, useDagger);
    D_eo(vec + l.vol/2, res, useDagger);
    D_oe(vec , res + l.vol/2, useDagger);
}


void DiracOP::applyDhatTo(spinor_iter vec, spinor_iter res, MatrixType const useDagger){
    
    std::vector<vec_fc> temp(l.vol/2), temp2(l.vol/2);

    for(int i=0; i<l.vol/2; i++) {temp[i].setZero(); temp2[i].setZero(); res[i].setZero();}

    // for the useDagger consider inverting the order of the product
    D_ee(vec, res, useDagger);
    D_oe(vec, temp.begin(), useDagger);  
    D_oo_inv(temp.begin(), temp2.begin(), useDagger);
    for(int i=0; i<l.vol/2; i++) {temp[i].setZero();}
    D_eo(temp2.begin(), temp.begin(), useDagger);

    for(int i=0; i<l.vol/2; i++) res[i] -= temp[i];
}

void DiracOP::D_oo_inv(spinor_iter y, spinor_iter x, MatrixType const useDagger){
    std::complex<double> sigma;
    mat Minv;

    int const vol = l.vol/2;

    for(int i=0; i<vol; i++){
        sigma = 0.5 * (Pauli.tau0*mesons.M[i+l.vol/2]).trace();
        
        Minv(0,0) = mesons.g*mesons.M[i+l.vol/2](1,1) + 2.0 + M;
        Minv(1,1) = mesons.g*mesons.M[i+l.vol/2](0,0) + 2.0 + M;
        Minv(0,1) = - mesons.g*mesons.M[i+l.vol/2](0,1);
        Minv(1,0) = - mesons.g*mesons.M[i+l.vol/2](1,0);
        Minv /= Minv.determinant();

        sigma = 0.5 * (Minv(0,0) + Minv(1,1)); // this is actually sigma + m + 2

        if (useDagger == MatrixType::Dagger){
            x[i][0] =  im * ( conj((Minv(0,0) - sigma))    * y[i][1] +         conj(Minv(1,0))     * y[i][3] ) + sigma * y[i][0];
            x[i][1] = -im * ( conj((Minv(0,0) - sigma))    * y[i][0] +         conj(Minv(1,0))     * y[i][2] ) + sigma * y[i][1];
            x[i][2] =  im * (          conj(Minv(0,1))     * y[i][1] + conj((Minv(1,1) - sigma))   * y[i][3] ) + sigma * y[i][2];
            x[i][3] = -im * (          conj(Minv(0,1))     * y[i][0] + conj((Minv(1,1) - sigma))   * y[i][2] ) + sigma * y[i][3];
        } else {
            x[i][0] =  im * ( (Minv(0,0) - sigma) * y[i][1] +      Minv(0,1)       * y[i][3] ) + sigma * y[i][0];
            x[i][1] = -im * ( (Minv(0,0) - sigma) * y[i][0] +      Minv(0,1)       * y[i][2] ) + sigma * y[i][1];
            x[i][2] =  im * (       Minv(1,0)     * y[i][1] + (Minv(1,1) - sigma)  * y[i][3] ) + sigma * y[i][2];
            x[i][3] = -im * (       Minv(1,0)     * y[i][0] + (Minv(1,1) - sigma)  * y[i][2] ) + sigma * y[i][3];
        }
    }
}

void DiracOP::D_ee_inv(spinor_iter y, spinor_iter x, MatrixType const useDagger){
    std::complex<double> sigma;
    mat Minv;

    int const vol = l.vol/2;

    for(int i=0; i<vol; i++){
        sigma = 0.5 * (Pauli.tau0*mesons.M[i]).trace();
        
        Minv(0,0) = mesons.g*mesons.M[i](1,1) + 2.0 + M;
        Minv(1,1) = mesons.g*mesons.M[i](0,0) + 2.0 + M;
        Minv(0,1) = - mesons.g*mesons.M[i](0,1);
        Minv(1,0) = - mesons.g*mesons.M[i](1,0);
        Minv /= Minv.determinant();

        sigma = 0.5 * (Minv(0,0) + Minv(1,1)); // this is actually g sigma + m + 2

        if (useDagger == MatrixType::Dagger){
            x[i][0] =  im * ( conj((Minv(0,0) - sigma))    * y[i][1] +         conj(Minv(1,0))     * y[i][3] ) + sigma * y[i][0];
            x[i][1] = -im * ( conj((Minv(0,0) - sigma))    * y[i][0] +         conj(Minv(1,0))     * y[i][2] ) + sigma * y[i][1];
            x[i][2] =  im * (          conj(Minv(0,1))     * y[i][1] + conj((Minv(1,1) - sigma))   * y[i][3] ) + sigma * y[i][2];
            x[i][3] = -im * (          conj(Minv(0,1))     * y[i][0] + conj((Minv(1,1) - sigma))   * y[i][2] ) + sigma * y[i][3];
        } else {
            x[i][0] =  im * ( (Minv(0,0) - sigma) * y[i][1] +      Minv(0,1)       * y[i][3] ) + sigma * y[i][0];
            x[i][1] = -im * ( (Minv(0,0) - sigma) * y[i][0] +      Minv(0,1)       * y[i][2] ) + sigma * y[i][1];
            x[i][2] =  im * (       Minv(1,0)     * y[i][1] + (Minv(1,1) - sigma)  * y[i][3] ) + sigma * y[i][2];
            x[i][3] = -im * (       Minv(1,0)     * y[i][0] + (Minv(1,1) - sigma)  * y[i][2] ) + sigma * y[i][3];
        }
    }
}

void DiracOP::D_ee(spinor_iter y, spinor_iter x, MatrixType const useDagger){

    std::complex<double> sigma;
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        sigma = 0.5 * (Pauli.tau0*mesons.M[i]).trace();        
        if (useDagger == MatrixType::Dagger){
            x[i][0] +=  im * mesons.g * ( conj((mesons.M[i](0,0) - sigma)) * y[i][1] +         conj(mesons.M[i](1,0))     * y[i][3] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i][0];
            x[i][1] += -im * mesons.g * ( conj((mesons.M[i](0,0) - sigma)) * y[i][0] +         conj(mesons.M[i](1,0))     * y[i][2] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i][1];
            x[i][2] +=  im * mesons.g * (       conj(mesons.M[i](0,1))     * y[i][1] + conj((mesons.M[i](1,1) - sigma))   * y[i][3] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i][2];
            x[i][3] += -im * mesons.g * (       conj(mesons.M[i](0,1))     * y[i][0] + conj((mesons.M[i](1,1) - sigma))   * y[i][2] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i][3];
        } else {
            x[i][0] +=  im * mesons.g * ( (mesons.M[i](0,0) - sigma) * y[i][1] +         mesons.M[i](0,1)     * y[i][3] ) + (2.0 + M + mesons.g*sigma) * y[i][0];
            x[i][1] += -im * mesons.g * ( (mesons.M[i](0,0) - sigma) * y[i][0] +         mesons.M[i](0,1)     * y[i][2] ) + (2.0 + M + mesons.g*sigma) * y[i][1];
            x[i][2] +=  im * mesons.g * (       mesons.M[i](1,0)     * y[i][1] + (mesons.M[i](1,1) - sigma)   * y[i][3] ) + (2.0 + M + mesons.g*sigma) * y[i][2];
            x[i][3] += -im * mesons.g * (       mesons.M[i](1,0)     * y[i][0] + (mesons.M[i](1,1) - sigma)   * y[i][2] ) + (2.0 + M + mesons.g*sigma) * y[i][3];
        }

    }

}

void DiracOP::D_oo(spinor_iter y, spinor_iter x, MatrixType const useDagger){

    std::complex<double> sigma;
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        sigma = 0.5 * (Pauli.tau0*mesons.M[i+vol]).trace();
        if (useDagger == MatrixType::Dagger){
            x[i][0] +=  im * mesons.g * ( conj((mesons.M[i+vol](0,0) - sigma)) * y[i][1] +         conj(mesons.M[i+vol](1,0))     * y[i][3] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i][0];
            x[i][1] += -im * mesons.g * ( conj((mesons.M[i+vol](0,0) - sigma)) * y[i][0] +         conj(mesons.M[i+vol](1,0))     * y[i][2] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i][1];
            x[i][2] +=  im * mesons.g * (       conj(mesons.M[i+vol](0,1))     * y[i][1] + conj((mesons.M[i+vol](1,1) - sigma))   * y[i][3] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i][2];
            x[i][3] += -im * mesons.g * (       conj(mesons.M[i+vol](0,1))     * y[i][0] + conj((mesons.M[i+vol](1,1) - sigma))   * y[i][2] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i][3];
        } else{
            x[i][0] +=  im * mesons.g * ( (mesons.M[i+vol](0,0) - sigma) * y[i][1] +         mesons.M[i+vol](0,1)     * y[i][3] ) + (2.0 + M + mesons.g*sigma) * y[i][0];
            x[i][1] += -im * mesons.g * ( (mesons.M[i+vol](0,0) - sigma) * y[i][0] +         mesons.M[i+vol](0,1)     * y[i][2] ) + (2.0 + M + mesons.g*sigma) * y[i][1];
            x[i][2] +=  im * mesons.g * (       mesons.M[i+vol](1,0)     * y[i][1] + (mesons.M[i+vol](1,1) - sigma)   * y[i][3] ) + (2.0 + M + mesons.g*sigma) * y[i][2];
            x[i][3] += -im * mesons.g * (       mesons.M[i+vol](1,0)     * y[i][0] + (mesons.M[i+vol](1,1) - sigma)   * y[i][2] ) + (2.0 + M + mesons.g*sigma) * y[i][3];
        }

    }
}


void DiracOP::D_eo(spinor_iter y, spinor_iter x, MatrixType const useDagger){

    std::vector<int> idx(2);
    int const vol = l.vol/2;


    int const Nt=l.Nt, Nx=l.Nx;

    double sgn[2];
    int nt, nx;
    
    for(int i=0; i<vol; i++){

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<double> psisum[2], psidiff[2];
        
        if (useDagger == MatrixType::Dagger) {

            psisum[0] = y[l.IDN[i][1] - vol][0] + y[l.IDN[i][1] - vol][1];
            psisum[1] = y[l.IDN[i][1] - vol][2] + y[l.IDN[i][1] - vol][3];
            psidiff[0] = y[l.IUP[i][1] - vol][0] - y[l.IUP[i][1] - vol][1];
            psidiff[1] = y[l.IUP[i][1] - vol][2] - y[l.IUP[i][1] - vol][3];

            x[l.toEOflat(nt, nx)][0] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol][0] + 0.5*psidiff[0] + 0.5*psisum[0];
            x[l.toEOflat(nt, nx)][2] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol][2] + 0.5*psidiff[1] + 0.5*psisum[1];
            x[l.toEOflat(nt, nx)][1] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol][1] - 0.5*psidiff[0] + 0.5*psisum[0];
            x[l.toEOflat(nt, nx)][3] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol][3] - 0.5*psidiff[1] + 0.5*psisum[1];

        } else {

            psisum[0] = y[l.IUP[i][1] - vol][0] + y[l.IUP[i][1] - vol][1];
            psisum[1] = y[l.IUP[i][1] - vol][2] + y[l.IUP[i][1] - vol][3];
            psidiff[0] = y[l.IDN[i][1] - vol][0] - y[l.IDN[i][1] - vol][1];
            psidiff[1] = y[l.IDN[i][1] - vol][2] - y[l.IDN[i][1] - vol][3];

            x[l.toEOflat(nt, nx)][0] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol][0] + 0.5*psisum[0] + 0.5*psidiff[0];
            x[l.toEOflat(nt, nx)][2] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol][2] + 0.5*psisum[1] + 0.5*psidiff[1];
            x[l.toEOflat(nt, nx)][1] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol][1] + 0.5*psisum[0] - 0.5*psidiff[0];
            x[l.toEOflat(nt, nx)][3] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol][3] + 0.5*psisum[1] - 0.5*psidiff[1];

        }
                                            
    }

}
void DiracOP::D_oe(spinor_iter y, spinor_iter x, MatrixType const useDagger){
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Hopping term
    int const Nt=l.Nt, Nx=l.Nx;

    double sgn[2];
    int nt, nx;

    int i;
    for(int j=0; j<vol; j++){

        i = j+vol; // full index (for lookup tables)

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<double> psisum[2], psidiff[2];
        
        if (useDagger == MatrixType::Dagger) {

            psisum[0] = y[l.IDN[i][1]][0] + y[l.IDN[i][1]][1];
            psisum[1] = y[l.IDN[i][1]][2] + y[l.IDN[i][1]][3];
            psidiff[0] = y[l.IUP[i][1]][0] - y[l.IUP[i][1]][1];
            psidiff[1] = y[l.IUP[i][1]][2] - y[l.IUP[i][1]][3];

            x[j][0] -=  sgn[1] * 1.0 * y[l.IDN[i][0]][0] + 0.5*(psidiff[0] + psisum[0]);
            x[j][2] -=  sgn[1] * 1.0 * y[l.IDN[i][0]][2] + 0.5*(psidiff[1] + psisum[1]);
            x[j][1] -=  sgn[0] * 1.0 * y[l.IUP[i][0]][1] - 0.5*(psidiff[0] - psisum[0]);
            x[j][3] -=  sgn[0] * 1.0 * y[l.IUP[i][0]][3] - 0.5*(psidiff[1] - psisum[1]);

        } else {

            psisum[0] = y[l.IUP[i][1]][0] + y[l.IUP[i][1]][1];
            psisum[1] = y[l.IUP[i][1]][2] + y[l.IUP[i][1]][3];
            psidiff[0] = y[l.IDN[i][1]][0] - y[l.IDN[i][1]][1];
            psidiff[1] = y[l.IDN[i][1]][2] - y[l.IDN[i][1]][3];

            x[j][0] -=  sgn[0] * 1.0 * y[l.IUP[i][0]][0] + 0.5*psisum[0] + 0.5*psidiff[0];
            x[j][2] -=  sgn[0] * 1.0 * y[l.IUP[i][0]][2] + 0.5*psisum[1] + 0.5*psidiff[1];
            x[j][1] -=  sgn[1] * 1.0 * y[l.IDN[i][0]][1] + 0.5*psisum[0] - 0.5*psidiff[0];
            x[j][3] -=  sgn[1] * 1.0 * y[l.IDN[i][0]][3] + 0.5*psisum[1] - 0.5*psidiff[1];


        }
                                            
    }
}



/*
void DiracOP::applyDhatTo_single(spinor_single_iter vec, spinor_single_iter res, MatrixType const useDagger=MatrixType::Normal){
    assert(useDagger==0 || useDagger==1);
    
    std::vector<vec_fc_single> temp(l.vol/2), temp2(l.vol/2);

    for(int i=0; i<l.vol/2; i++) {temp[i].setZero(); temp2[i].setZero(); res[i].setZero();}

    // for the useDagger consider inverting the order of the product
    D_ee_single(vec, res, useDagger);
    D_oe_single(vec, temp.begin(), useDagger);  
    D_oo_inv_single(temp.begin(), temp2.begin(), useDagger);
    for(int i=0; i<l.vol/2; i++) {temp[i].setZero();}
    D_eo_single(temp2.begin(), temp.begin(), useDagger);

    for(int i=0; i<l.vol/2; i++) res[i] -= temp[i];
}

void DiracOP::D_ee_single(spinor_single_iter y, spinor_single_iter x, MatrixType const useDagger){

    std::complex<float> sigma;
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    std::complex<float> m = (std::complex<float>) this->M;
    std::complex<float> g = (std::complex<float>) mesons.g;
    std::complex<float> constexpr oneHalf = 0.5;

    // Diagonal term
    for(int i=0; i<vol; i++){
        sigma = oneHalf * (Pauli.tau0.cast<std::complex<float>>()*mesons.M_single[i]).trace().real();
        x[i] += (std::complex<float>{2., 0.} + m + g*sigma) * y[i];
        if (useDagger == MatrixType::Dagger)
            x[i] += buildCompositeOP_single(g * (mesons.M_single[i] - sigma*mat_single::Identity()).adjoint(), gamma5.cast<std::complex<float>>()) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            x[i] += buildCompositeOP_single(g * (mesons.M_single[i] - sigma*mat_single::Identity()), gamma5.cast<std::complex<float>>()) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)

    }

}

void DiracOP::D_oo_single(spinor_single_iter y, spinor_single_iter x, MatrixType const useDagger){
    float sigma;
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        idx = l.eoToVec(i+vol);
        sigma = 0.5 * (Pauli.tau0.cast<std::complex<float>>()*mesons.M_single[i+vol]).trace().real();
        x[i] += (2.0 + M + mesons.g*sigma) * y[i];
        if (useDagger == MatrixType::Dagger)
            x[i] += buildCompositeOP_single(mesons.g * (mesons.M_single[i+vol] - sigma*mat_single::Identity()).adjoint(), gamma5.cast<std::complex<float>>()) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            x[i] += buildCompositeOP_single(mesons.g * (mesons.M_single[i+vol] - sigma*mat_single::Identity()), gamma5.cast<std::complex<float>>()) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)

    }
}


void DiracOP::D_eo_single(spinor_single_iter y, spinor_single_iter x, MatrixType const useDagger){

    std::vector<int> idx(2);
    int const vol = l.vol/2;
    int const Nt=l.Nt, Nx=l.Nx;

    std::complex<float> sgn[2];
    int nt, nx;
    std::complex<float> psisum[2], psidiff[2];
    std::complex<float> c {0.5, 0.};
    
    for(int i=0; i<vol; i++){

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        // maybe use ? : syntax to remove the if (more elegant?)
        
        if (useDagger == MatrixType::Dagger) {

            psisum[0] = y[l.IDN[i][1] - vol][0] + y[l.IDN[i][1] - vol][1];
            psisum[1] = y[l.IDN[i][1] - vol][2] + y[l.IDN[i][1] - vol][3];
            psidiff[0] = y[l.IUP[i][1] - vol][0] - y[l.IUP[i][1] - vol][1];
            psidiff[1] = y[l.IUP[i][1] - vol][2] - y[l.IUP[i][1] - vol][3];

            x[l.toEOflat(nt, nx)][0] -=  sgn[1] * y[l.IDN[i][0] - vol][0] + c*psidiff[0] + c*psisum[0];
            x[l.toEOflat(nt, nx)][2] -=  sgn[1] * y[l.IDN[i][0] - vol][2] + c*psidiff[1] + c*psisum[1];
            x[l.toEOflat(nt, nx)][1] -=  sgn[0] * y[l.IUP[i][0] - vol][1] - c*psidiff[0] + c*psisum[0];
            x[l.toEOflat(nt, nx)][3] -=  sgn[0] * y[l.IUP[i][0] - vol][3] - c*psidiff[1] + c*psisum[1];

        } else {

            psisum[0] = y[l.IUP[i][1] - vol][0] + y[l.IUP[i][1] - vol][1];
            psisum[1] = y[l.IUP[i][1] - vol][2] + y[l.IUP[i][1] - vol][3];
            psidiff[0] = y[l.IDN[i][1] - vol][0] - y[l.IDN[i][1] - vol][1];
            psidiff[1] = y[l.IDN[i][1] - vol][2] - y[l.IDN[i][1] - vol][3];

            x[l.toEOflat(nt, nx)][0] -=  sgn[0] * y[l.IUP[i][0] - vol][0] + c*psisum[0] + c*psidiff[0];
            x[l.toEOflat(nt, nx)][2] -=  sgn[0] * y[l.IUP[i][0] - vol][2] + c*psisum[1] + c*psidiff[1];
            x[l.toEOflat(nt, nx)][1] -=  sgn[1] * y[l.IDN[i][0] - vol][1] + c*psisum[0] - c*psidiff[0];
            x[l.toEOflat(nt, nx)][3] -=  sgn[1] * y[l.IDN[i][0] - vol][3] + c*psisum[1] - c*psidiff[1];

        }
                                            
    }

}
void DiracOP::D_oe_single(spinor_single_iter y, spinor_single_iter x, MatrixType const useDagger){
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Hopping term
    int const Nt=l.Nt, Nx=l.Nx;

    float sgn[2];
    int nt, nx;
    int i;
    std::complex<float> psisum[2], psidiff[2];
    std::complex<float> c {0.5, 0.};

    for(int j=0; j<vol; j++){

        i = j+vol; // full index (for lookup tables)

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        
        if (useDagger == MatrixType::Dagger) {

            psisum[0] = y[l.IDN[i][1]][0] + y[l.IDN[i][1]][1];
            psisum[1] = y[l.IDN[i][1]][2] + y[l.IDN[i][1]][3];
            psidiff[0] = y[l.IUP[i][1]][0] - y[l.IUP[i][1]][1];
            psidiff[1] = y[l.IUP[i][1]][2] - y[l.IUP[i][1]][3];

            x[j][0] -=  sgn[1] * y[l.IDN[i][0]][0] + c*psidiff[0] + c*psisum[0];
            x[j][2] -=  sgn[1] * y[l.IDN[i][0]][2] + c*psidiff[1] + c*psisum[1];
            x[j][1] -=  sgn[0] * y[l.IUP[i][0]][1] - c*psidiff[0] + c*psisum[0];
            x[j][3] -=  sgn[0] * y[l.IUP[i][0]][3] - c*psidiff[1] + c*psisum[1];

        } else {

            psisum[0] = y[l.IUP[i][1]][0] + y[l.IUP[i][1]][1];
            psisum[1] = y[l.IUP[i][1]][2] + y[l.IUP[i][1]][3];
            psidiff[0] = y[l.IDN[i][1]][0] - y[l.IDN[i][1]][1];
            psidiff[1] = y[l.IDN[i][1]][2] - y[l.IDN[i][1]][3];

            x[j][0] -=  sgn[0] * y[l.IUP[i][0]][0] + c*psisum[0] + c*psidiff[0];
            x[j][2] -=  sgn[0] * y[l.IUP[i][0]][2] + c*psisum[1] + c*psidiff[1];
            x[j][1] -=  sgn[1] * y[l.IDN[i][0]][1] + c*psisum[0] - c*psidiff[0];
            x[j][3] -=  sgn[1] * y[l.IDN[i][0]][3] + c*psisum[1] - c*psidiff[1];


        }
                                            
    }
}

void DiracOP::applyTo_single(spinor_single_iter vec, spinor_single_iter res, MatrixType const useDagger){
    for(int i=0; i<l.vol; i++) {res[i].setZero();}
    D_ee_single(vec, res, useDagger);
    D_oo_single(vec + l.vol/2, res + l.vol/2, useDagger);
    D_eo_single(vec + l.vol/2, res, useDagger);
    D_oe_single(vec , res + l.vol/2, useDagger);
}

*/

template void DiracOP::applyTo<spinor_iter>(spinor_iter vec, spinor_iter res, MatrixType const useDagger);
template void DiracOP::applyTo<spinor_single_iter>(spinor_single_iter vec, spinor_single_iter res, MatrixType const useDagger);


