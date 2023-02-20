#include "DiracOP.h"

// maybe bring setZero() inside Dee Doo, ...

DiracOP::DiracOP(double const M, O4Mesons& mesons, Lattice& l) : 
    M{M}, 
    mesons{mesons},
    l{l}
    {;}

template <typename T>
void DiracOP::applyTo(T vec, T res, MatrixType const useDagger){
    std::fill(res, res + l.vol, Spinor());
    D_ee(vec, res, useDagger);
    D_oo(vec + l.vol/2, res + l.vol/2, useDagger);
    D_eo(vec + l.vol/2, res, useDagger);
    D_oe(vec , res + l.vol/2, useDagger);
}


void DiracOP::applyDhatTo(vecfield_iter vec, vecfield_iter res, MatrixType const useDagger){
    
    vecfield temp(l.vol/2, Spinor()), temp2(l.vol/2, Spinor());
    std::fill(res, res + l.vol/2, Spinor());


    // for the useDagger consider inverting the order of the product
    D_ee(vec, res, useDagger);
    D_oe(vec, temp.begin(), useDagger);  
    D_oo_inv(temp.begin(), temp2.begin(), useDagger);
    std::fill(temp.begin(), temp.begin() + l.vol/2, Spinor());
    D_eo(temp2.begin(), temp.begin(), useDagger);

    for(int i=0; i<l.vol/2; i++) { for (int j=0; j<4; j++) res[i].val[j] -= temp[i].val[j]; }
}

void DiracOP::D_oo_inv(vecfield_iter y, vecfield_iter x, MatrixType const useDagger){
    std::complex<double> sigma, det;
    O4Mat Minv;

    int const vol = l.vol/2;

    for(int i=0; i<vol; i++){
        
        Minv.val[0][0] = mesons.g*mesons.M[i+l.vol/2].val[1][1] + 2.0 + M;
        Minv.val[1][1] = mesons.g*mesons.M[i+l.vol/2].val[0][0] + 2.0 + M;
        Minv.val[0][1] = - mesons.g*mesons.M[i+l.vol/2].val[0][1];
        Minv.val[1][0] = - mesons.g*mesons.M[i+l.vol/2].val[1][0];
        det = Minv.determinant();
        Minv.val[0][0] /= det;
        Minv.val[0][1] /= det; 
        Minv.val[1][0] /= det; 
        Minv.val[1][1] /= det; 


        sigma = 0.5 * (Minv.val[0][0] + Minv.val[1][1]); // this is actually sigma + m + 2

        if (useDagger == MatrixType::Dagger){
            x[i].val[0] =  im * ( conj((Minv.val[0][0] - sigma))    * y[i].val[1] +         conj(Minv.val[1][0])     * y[i].val[3] ) + sigma * y[i].val[0];
            x[i].val[1] = -im * ( conj((Minv.val[0][0] - sigma))    * y[i].val[0] +         conj(Minv.val[1][0])     * y[i].val[2] ) + sigma * y[i].val[1];
            x[i].val[2] =  im * (          conj(Minv.val[0][1])     * y[i].val[1] + conj((Minv.val[1][1] - sigma))   * y[i].val[3] ) + sigma * y[i].val[2];
            x[i].val[3] = -im * (          conj(Minv.val[0][1])     * y[i].val[0] + conj((Minv.val[1][1] - sigma))   * y[i].val[2] ) + sigma * y[i].val[3];
        } else {
            x[i].val[0] =  im * ( (Minv.val[0][0] - sigma) * y[i].val[1] +      Minv.val[0][1]       * y[i].val[3] ) + sigma * y[i].val[0];
            x[i].val[1] = -im * ( (Minv.val[0][0] - sigma) * y[i].val[0] +      Minv.val[0][1]       * y[i].val[2] ) + sigma * y[i].val[1];
            x[i].val[2] =  im * (       Minv.val[1][0]     * y[i].val[1] + (Minv.val[1][1] - sigma)  * y[i].val[3] ) + sigma * y[i].val[2];
            x[i].val[3] = -im * (       Minv.val[1][0]     * y[i].val[0] + (Minv.val[1][1] - sigma)  * y[i].val[2] ) + sigma * y[i].val[3];
        }
    }
}

void DiracOP::D_ee_inv(vecfield_iter y, vecfield_iter x, MatrixType const useDagger){
    std::complex<double> sigma, det;
    O4Mat Minv;

    int const vol = l.vol/2;

    for(int i=0; i<vol; i++){
        
        Minv.val[0][0] = mesons.g*mesons.M[i].val[1][1] + 2.0 + M;
        Minv.val[1][1] = mesons.g*mesons.M[i].val[0][0] + 2.0 + M;
        Minv.val[0][1] = - mesons.g*mesons.M[i].val[0][1];
        Minv.val[1][0] = - mesons.g*mesons.M[i].val[1][0];
        det = Minv.determinant();
        Minv.val[0][0] /= det;
        Minv.val[0][1] /= det; 
        Minv.val[1][0] /= det; 
        Minv.val[1][1] /= det; 

        sigma = 0.5 * (Minv.val[0][0] + Minv.val[1][1]); // this is actually g sigma + m + 2

        if (useDagger == MatrixType::Dagger){
            x[i].val[0] =  im * ( conj((Minv.val[0][0] - sigma))    * y[i].val[1] +         conj(Minv.val[1][0])     * y[i].val[3] ) + sigma * y[i].val[0];
            x[i].val[1] = -im * ( conj((Minv.val[0][0] - sigma))    * y[i].val[0] +         conj(Minv.val[1][0])     * y[i].val[2] ) + sigma * y[i].val[1];
            x[i].val[2] =  im * (          conj(Minv.val[0][1])     * y[i].val[1] + conj((Minv.val[1][1] - sigma))   * y[i].val[3] ) + sigma * y[i].val[2];
            x[i].val[3] = -im * (          conj(Minv.val[0][1])     * y[i].val[0] + conj((Minv.val[1][1] - sigma))   * y[i].val[2] ) + sigma * y[i].val[3];
        } else {
            x[i].val[0] =  im * ( (Minv.val[0][0] - sigma) * y[i].val[1] +      Minv.val[0][1]       * y[i].val[3] ) + sigma * y[i].val[0];
            x[i].val[1] = -im * ( (Minv.val[0][0] - sigma) * y[i].val[0] +      Minv.val[0][1]       * y[i].val[2] ) + sigma * y[i].val[1];
            x[i].val[2] =  im * (       Minv.val[1][0]     * y[i].val[1] + (Minv.val[1][1] - sigma)  * y[i].val[3] ) + sigma * y[i].val[2];
            x[i].val[3] = -im * (       Minv.val[1][0]     * y[i].val[0] + (Minv.val[1][1] - sigma)  * y[i].val[2] ) + sigma * y[i].val[3];
        }
    }
}

void DiracOP::D_ee(vecfield_iter y, vecfield_iter x, MatrixType const useDagger){

    std::complex<double> sigma;
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        sigma = 0.5 * (mesons.M[i].val[0][0] + mesons.M[i].val[1][1]);        
        if (useDagger == MatrixType::Dagger){
            x[i].val[0] +=  im * mesons.g * ( conj((mesons.M[i].val[0][0] - sigma)) * y[i].val[1] +         conj(mesons.M[i].val[1][0])     * y[i].val[3] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i].val[0];
            x[i].val[1] += -im * mesons.g * ( conj((mesons.M[i].val[0][0] - sigma)) * y[i].val[0] +         conj(mesons.M[i].val[1][0])     * y[i].val[2] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i].val[1];
            x[i].val[2] +=  im * mesons.g * (       conj(mesons.M[i].val[0][1])     * y[i].val[1] + conj((mesons.M[i].val[1][1] - sigma))   * y[i].val[3] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i].val[2];
            x[i].val[3] += -im * mesons.g * (       conj(mesons.M[i].val[0][1])     * y[i].val[0] + conj((mesons.M[i].val[1][1] - sigma))   * y[i].val[2] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i].val[3];
        } else {
            x[i].val[0] +=  im * mesons.g * ( (mesons.M[i].val[0][0] - sigma) * y[i].val[1] +         mesons.M[i].val[0][1]     * y[i].val[3] ) + (2.0 + M + mesons.g*sigma) * y[i].val[0];
            x[i].val[1] += -im * mesons.g * ( (mesons.M[i].val[0][0] - sigma) * y[i].val[0] +         mesons.M[i].val[0][1]     * y[i].val[2] ) + (2.0 + M + mesons.g*sigma) * y[i].val[1];
            x[i].val[2] +=  im * mesons.g * (       mesons.M[i].val[1][0]     * y[i].val[1] + (mesons.M[i].val[1][1] - sigma)   * y[i].val[3] ) + (2.0 + M + mesons.g*sigma) * y[i].val[2];
            x[i].val[3] += -im * mesons.g * (       mesons.M[i].val[1][0]     * y[i].val[0] + (mesons.M[i].val[1][1] - sigma)   * y[i].val[2] ) + (2.0 + M + mesons.g*sigma) * y[i].val[3];
        }

    }

}

void DiracOP::D_oo(vecfield_iter y, vecfield_iter x, MatrixType const useDagger){

    std::complex<double> sigma;
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        sigma = 0.5 * (mesons.M[i+l.vol/2].val[0][0] + mesons.M[i+l.vol/2].val[1][1]);
        if (useDagger == MatrixType::Dagger){
            x[i].val[0] +=  im * mesons.g * ( conj((mesons.M[i+vol].val[0][0] - sigma)) * y[i].val[1] +         conj(mesons.M[i+vol].val[1][0])     * y[i].val[3] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i].val[0];
            x[i].val[1] += -im * mesons.g * ( conj((mesons.M[i+vol].val[0][0] - sigma)) * y[i].val[0] +         conj(mesons.M[i+vol].val[1][0])     * y[i].val[2] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i].val[1];
            x[i].val[2] +=  im * mesons.g * (       conj(mesons.M[i+vol].val[0][1])     * y[i].val[1] + conj((mesons.M[i+vol].val[1][1] - sigma))   * y[i].val[3] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i].val[2];
            x[i].val[3] += -im * mesons.g * (       conj(mesons.M[i+vol].val[0][1])     * y[i].val[0] + conj((mesons.M[i+vol].val[1][1] - sigma))   * y[i].val[2] ) + (2.0 + M + mesons.g*conj(sigma)) * y[i].val[3];
        } else{
            x[i].val[0] +=  im * mesons.g * ( (mesons.M[i+vol].val[0][0] - sigma) * y[i].val[1] +         mesons.M[i+vol].val[0][1]     * y[i].val[3] ) + (2.0 + M + mesons.g*sigma) * y[i].val[0];
            x[i].val[1] += -im * mesons.g * ( (mesons.M[i+vol].val[0][0] - sigma) * y[i].val[0] +         mesons.M[i+vol].val[0][1]     * y[i].val[2] ) + (2.0 + M + mesons.g*sigma) * y[i].val[1];
            x[i].val[2] +=  im * mesons.g * (       mesons.M[i+vol].val[1][0]     * y[i].val[1] + (mesons.M[i+vol].val[1][1] - sigma)   * y[i].val[3] ) + (2.0 + M + mesons.g*sigma) * y[i].val[2];
            x[i].val[3] += -im * mesons.g * (       mesons.M[i+vol].val[1][0]     * y[i].val[0] + (mesons.M[i+vol].val[1][1] - sigma)   * y[i].val[2] ) + (2.0 + M + mesons.g*sigma) * y[i].val[3];
        }

    }
}


void DiracOP::D_eo(vecfield_iter y, vecfield_iter x, MatrixType const useDagger){

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

            psisum[0]  = y[l.IDN[i][1] - vol].val[0] + y[l.IDN[i][1] - vol].val[1];
            psisum[1]  = y[l.IDN[i][1] - vol].val[2] + y[l.IDN[i][1] - vol].val[3];
            psidiff[0] = y[l.IUP[i][1] - vol].val[0] - y[l.IUP[i][1] - vol].val[1];
            psidiff[1] = y[l.IUP[i][1] - vol].val[2] - y[l.IUP[i][1] - vol].val[3];

            x[l.toEOflat(nt, nx)].val[0] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol].val[0] + 0.5*psidiff[0] + 0.5*psisum[0];
            x[l.toEOflat(nt, nx)].val[2] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol].val[2] + 0.5*psidiff[1] + 0.5*psisum[1];
            x[l.toEOflat(nt, nx)].val[1] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol].val[1] - 0.5*psidiff[0] + 0.5*psisum[0];
            x[l.toEOflat(nt, nx)].val[3] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol].val[3] - 0.5*psidiff[1] + 0.5*psisum[1];

        } else {

            psisum[0]  = y[l.IUP[i][1] - vol].val[0] + y[l.IUP[i][1] - vol].val[1];
            psisum[1]  = y[l.IUP[i][1] - vol].val[2] + y[l.IUP[i][1] - vol].val[3];
            psidiff[0] = y[l.IDN[i][1] - vol].val[0] - y[l.IDN[i][1] - vol].val[1];
            psidiff[1] = y[l.IDN[i][1] - vol].val[2] - y[l.IDN[i][1] - vol].val[3];

            x[l.toEOflat(nt, nx)].val[0] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol].val[0] + 0.5*psisum[0] + 0.5*psidiff[0];
            x[l.toEOflat(nt, nx)].val[2] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol].val[2] + 0.5*psisum[1] + 0.5*psidiff[1];
            x[l.toEOflat(nt, nx)].val[1] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol].val[1] + 0.5*psisum[0] - 0.5*psidiff[0];
            x[l.toEOflat(nt, nx)].val[3] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol].val[3] + 0.5*psisum[1] - 0.5*psidiff[1];

        }
                                            
    }

}
void DiracOP::D_oe(vecfield_iter y, vecfield_iter x, MatrixType const useDagger){
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

            psisum[0]  = y[l.IDN[i][1]].val[0] + y[l.IDN[i][1]].val[1];
            psisum[1]  = y[l.IDN[i][1]].val[2] + y[l.IDN[i][1]].val[3];
            psidiff[0] = y[l.IUP[i][1]].val[0] - y[l.IUP[i][1]].val[1];
            psidiff[1] = y[l.IUP[i][1]].val[2] - y[l.IUP[i][1]].val[3];

            x[j].val[0] -=  sgn[1] * 1.0 * y[l.IDN[i][0]].val[0] + 0.5*(psidiff[0] + psisum[0]);
            x[j].val[2] -=  sgn[1] * 1.0 * y[l.IDN[i][0]].val[2] + 0.5*(psidiff[1] + psisum[1]);
            x[j].val[1] -=  sgn[0] * 1.0 * y[l.IUP[i][0]].val[1] - 0.5*(psidiff[0] - psisum[0]);
            x[j].val[3] -=  sgn[0] * 1.0 * y[l.IUP[i][0]].val[3] - 0.5*(psidiff[1] - psisum[1]);

        } else {

            psisum[0]  = y[l.IUP[i][1]].val[0] + y[l.IUP[i][1]].val[1];
            psisum[1]  = y[l.IUP[i][1]].val[2] + y[l.IUP[i][1]].val[3];
            psidiff[0] = y[l.IDN[i][1]].val[0] - y[l.IDN[i][1]].val[1];
            psidiff[1] = y[l.IDN[i][1]].val[2] - y[l.IDN[i][1]].val[3];

            x[j].val[0] -=  sgn[0] * 1.0 * y[l.IUP[i][0]].val[0] + 0.5*psisum[0] + 0.5*psidiff[0];
            x[j].val[2] -=  sgn[0] * 1.0 * y[l.IUP[i][0]].val[2] + 0.5*psisum[1] + 0.5*psidiff[1];
            x[j].val[1] -=  sgn[1] * 1.0 * y[l.IDN[i][0]].val[1] + 0.5*psisum[0] - 0.5*psidiff[0];
            x[j].val[3] -=  sgn[1] * 1.0 * y[l.IDN[i][0]].val[3] + 0.5*psisum[1] - 0.5*psidiff[1];


        }
                                            
    }
}



/*
void DiracOP::applyDhatTo_single(vecfield_single_iter vec, vecfield_single_iter res, MatrixType const useDagger=MatrixType::Normal){
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

void DiracOP::D_ee_single(vecfield_single_iter y, vecfield_single_iter x, MatrixType const useDagger){

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

void DiracOP::D_oo_single(vecfield_single_iter y, vecfield_single_iter x, MatrixType const useDagger){
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


void DiracOP::D_eo_single(vecfield_single_iter y, vecfield_single_iter x, MatrixType const useDagger){

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
void DiracOP::D_oe_single(vecfield_single_iter y, vecfield_single_iter x, MatrixType const useDagger){
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

void DiracOP::applyTo_single(vecfield_single_iter vec, vecfield_single_iter res, MatrixType const useDagger){
    for(int i=0; i<l.vol; i++) {res[i].setZero();}
    D_ee_single(vec, res, useDagger);
    D_oo_single(vec + l.vol/2, res + l.vol/2, useDagger);
    D_eo_single(vec + l.vol/2, res, useDagger);
    D_oe_single(vec , res + l.vol/2, useDagger);
}

*/

template void DiracOP::applyTo<vecfield_iter>(vecfield_iter vec, vecfield_iter res, MatrixType const useDagger);
//template void DiracOP::applyTo<vecfield_single_iter>(vecfield_single_iter vec, vecfield_single_iter res, MatrixType const useDagger);


