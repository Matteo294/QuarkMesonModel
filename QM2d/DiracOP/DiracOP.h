#pragma once

#include <vector>
#include <complex>
#include "../SpinorField/SpinorField.h"
#include "../Mesons/O4Mesons.h"
#include "../functions/functions.h"

template <typename Titer, typename Tvar>
class DiracOP {
    public:
        DiracOP(double const M, O4Mesons& mesons, Lattice& l);
        ~DiracOP(){}
        
        void applyTo(Titer vec, Titer res, MatrixType const useDagger=MatrixType::Normal); // apply full Dirac operator to vector of dimension Nt*Nx
        void applyDhatTo(Titer vec, Titer res, MatrixType const useDagger=MatrixType::Normal); // apply Dhat = Dee - Deo Doo_inv Doe to a vector of dimension Nt*Nx/2

        // NB in the following functions MODIFY the vector passed !!
        void D_oo_inv(Titer inVec, Titer outVec, MatrixType const useDagger=MatrixType::Normal);
        void D_ee_inv(Titer inVec, Titer outVec, MatrixType const useDagger=MatrixType::Normal);
        void D_ee(Titer inVec, Titer outVec, MatrixType const useDagger=MatrixType::Normal);
        void D_oo(Titer inVec, Titer outVec, MatrixType const useDagger=MatrixType::Normal);
        void D_eo(Titer inVec, Titer outVec, MatrixType const useDagger=MatrixType::Normal);
        void D_oe(Titer inVec, Titer outVec, MatrixType const useDagger=MatrixType::Normal);

        void applyN(Titer inBegin, Titer inEnd, Titer outBegin);
        
    private:
        Lattice& l;
        O4Mesons& mesons;
        double const M;

};

typedef DiracOP<vecfield_iter, double> DiracOP_d;
typedef DiracOP<vecfield_single_iter, float> DiracOP_f;

template <typename Titer, typename Tvar>
void DiracOP<Titer, Tvar>::D_ee(Titer inVec, Titer outVec, MatrixType const useDagger){

    std::complex<Tvar> sigma;
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        sigma = 0.5 * (mesons.M[i].val[0][0] + mesons.M[i].val[1][1]);        
        if (useDagger == MatrixType::Dagger){
            outVec[i].val[0] +=  im * mesons.g * ( conj((mesons.M[i].val[0][0] - sigma)) * inVec[i].val[1] +         conj(mesons.M[i].val[1][0])     * inVec[i].val[3] ) + (2.0 + M + mesons.g*conj(sigma)) * inVec[i].val[0];
            outVec[i].val[1] += -im * mesons.g * ( conj((mesons.M[i].val[0][0] - sigma)) * inVec[i].val[0] +         conj(mesons.M[i].val[1][0])     * inVec[i].val[2] ) + (2.0 + M + mesons.g*conj(sigma)) * inVec[i].val[1];
            outVec[i].val[2] +=  im * mesons.g * (       conj(mesons.M[i].val[0][1])     * inVec[i].val[1] + conj((mesons.M[i].val[1][1] - sigma))   * inVec[i].val[3] ) + (2.0 + M + mesons.g*conj(sigma)) * inVec[i].val[2];
            outVec[i].val[3] += -im * mesons.g * (       conj(mesons.M[i].val[0][1])     * inVec[i].val[0] + conj((mesons.M[i].val[1][1] - sigma))   * inVec[i].val[2] ) + (2.0 + M + mesons.g*conj(sigma)) * inVec[i].val[3];
        } else {
            outVec[i].val[0] +=  im * mesons.g * ( (mesons.M[i].val[0][0] - sigma) * inVec[i].val[1] +         mesons.M[i].val[0][1]     * inVec[i].val[3] ) + (2.0 + M + mesons.g*sigma) * inVec[i].val[0];
            outVec[i].val[1] += -im * mesons.g * ( (mesons.M[i].val[0][0] - sigma) * inVec[i].val[0] +         mesons.M[i].val[0][1]     * inVec[i].val[2] ) + (2.0 + M + mesons.g*sigma) * inVec[i].val[1];
            outVec[i].val[2] +=  im * mesons.g * (       mesons.M[i].val[1][0]     * inVec[i].val[1] + (mesons.M[i].val[1][1] - sigma)   * inVec[i].val[3] ) + (2.0 + M + mesons.g*sigma) * inVec[i].val[2];
            outVec[i].val[3] += -im * mesons.g * (       mesons.M[i].val[1][0]     * inVec[i].val[0] + (mesons.M[i].val[1][1] - sigma)   * inVec[i].val[2] ) + (2.0 + M + mesons.g*sigma) * inVec[i].val[3];
        }

    }

}

template <typename Titer, typename Tvar>
DiracOP<Titer, Tvar>::DiracOP(double const M, O4Mesons& mesons, Lattice& l) : 
    M{M}, 
    mesons{mesons},
    l{l}
    {;}

template <typename Titer, typename Tvar>
void DiracOP<Titer, Tvar>::applyTo(Titer vec, Titer res, MatrixType const useDagger){
    std::fill(res, res + l.vol, Spinor<Tvar>());
    D_ee(vec, res, useDagger);
    D_oo(vec + l.vol/2, res + l.vol/2, useDagger);
    D_eo(vec + l.vol/2, res, useDagger);
    D_oe(vec , res + l.vol/2, useDagger);
}

template <typename Titer, typename Tvar>
void DiracOP<Titer, Tvar>::applyDhatTo(Titer vec, Titer res, MatrixType const useDagger){
    
    vecfield temp(l.vol/2, Spinor<Tvar>()), temp2(l.vol/2, Spinor<Tvar>());
    std::fill(res, res + l.vol/2, Spinor<Tvar>());


    // for the useDagger consider inverting the order of the product
    D_ee(vec, res, useDagger);
    D_oe(vec, temp.begin(), useDagger);  
    D_oo_inv(temp.begin(), temp2.begin(), useDagger);
    std::fill(temp.begin(), temp.begin() + l.vol/2, Spinor<Tvar>());
    D_eo(temp2.begin(), temp.begin(), useDagger);

    for(int i=0; i<l.vol/2; i++) { for (int j=0; j<4; j++) res[i].val[j] -= temp[i].val[j]; }
}

template <typename Titer, typename Tvar>
void DiracOP<Titer, Tvar>::D_oo_inv(Titer inVec, Titer outVec, MatrixType const useDagger){
    std::complex<Tvar> sigma, det;
    O4Mat<Tvar> Minv;

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
            outVec[i].val[0] +=  im * ( conj((Minv.val[0][0] - sigma))    * inVec[i].val[1] +         conj(Minv.val[1][0])     * inVec[i].val[3] ) + sigma * inVec[i].val[0];
            outVec[i].val[1] += -im * ( conj((Minv.val[0][0] - sigma))    * inVec[i].val[0] +         conj(Minv.val[1][0])     * inVec[i].val[2] ) + sigma * inVec[i].val[1];
            outVec[i].val[2] +=  im * (          conj(Minv.val[0][1])     * inVec[i].val[1] + conj((Minv.val[1][1] - sigma))   * inVec[i].val[3] ) + sigma * inVec[i].val[2];
            outVec[i].val[3] += -im * (          conj(Minv.val[0][1])     * inVec[i].val[0] + conj((Minv.val[1][1] - sigma))   * inVec[i].val[2] ) + sigma * inVec[i].val[3];
        } else {
            outVec[i].val[0] +=  im * ( (Minv.val[0][0] - sigma) * inVec[i].val[1] +      Minv.val[0][1]       * inVec[i].val[3] ) + sigma * inVec[i].val[0];
            outVec[i].val[1] += -im * ( (Minv.val[0][0] - sigma) * inVec[i].val[0] +      Minv.val[0][1]       * inVec[i].val[2] ) + sigma * inVec[i].val[1];
            outVec[i].val[2] +=  im * (       Minv.val[1][0]     * inVec[i].val[1] + (Minv.val[1][1] - sigma)  * inVec[i].val[3] ) + sigma * inVec[i].val[2];
            outVec[i].val[3] += -im * (       Minv.val[1][0]     * inVec[i].val[0] + (Minv.val[1][1] - sigma)  * inVec[i].val[2] ) + sigma * inVec[i].val[3];
        }
    }
}

template <typename Titer, typename Tvar>
void DiracOP<Titer, Tvar>::D_ee_inv(Titer inVec, Titer outVec, MatrixType const useDagger){
    std::complex<Tvar> sigma, det;
    O4Mat<Tvar> Minv;

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
            outVec[i].val[0] +=  im * ( conj((Minv.val[0][0] - sigma))    * inVec[i].val[1] +         conj(Minv.val[1][0])     * inVec[i].val[3] ) + sigma * inVec[i].val[0];
            outVec[i].val[1] += -im * ( conj((Minv.val[0][0] - sigma))    * inVec[i].val[0] +         conj(Minv.val[1][0])     * inVec[i].val[2] ) + sigma * inVec[i].val[1];
            outVec[i].val[2] +=  im * (          conj(Minv.val[0][1])     * inVec[i].val[1] + conj((Minv.val[1][1] - sigma))   * inVec[i].val[3] ) + sigma * inVec[i].val[2];
            outVec[i].val[3] += -im * (          conj(Minv.val[0][1])     * inVec[i].val[0] + conj((Minv.val[1][1] - sigma))   * inVec[i].val[2] ) + sigma * inVec[i].val[3];
        } else {
            outVec[i].val[0] +=  im * ( (Minv.val[0][0] - sigma) * inVec[i].val[1] +      Minv.val[0][1]       * inVec[i].val[3] ) + sigma * inVec[i].val[0];
            outVec[i].val[1] += -im * ( (Minv.val[0][0] - sigma) * inVec[i].val[0] +      Minv.val[0][1]       * inVec[i].val[2] ) + sigma * inVec[i].val[1];
            outVec[i].val[2] +=  im * (       Minv.val[1][0]     * inVec[i].val[1] + (Minv.val[1][1] - sigma)  * inVec[i].val[3] ) + sigma * inVec[i].val[2];
            outVec[i].val[3] += -im * (       Minv.val[1][0]     * inVec[i].val[0] + (Minv.val[1][1] - sigma)  * inVec[i].val[2] ) + sigma * inVec[i].val[3];
        }
    }
}

template <typename Titer, typename Tvar>
void DiracOP<Titer, Tvar>::D_oo(Titer inVec, Titer outVec, MatrixType const useDagger){

    std::complex<Tvar> sigma;
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        sigma = 0.5 * (mesons.M[i+l.vol/2].val[0][0] + mesons.M[i+l.vol/2].val[1][1]);
        if (useDagger == MatrixType::Dagger){
            outVec[i].val[0] +=  im * mesons.g * ( conj((mesons.M[i+vol].val[0][0] - sigma)) * inVec[i].val[1] +         conj(mesons.M[i+vol].val[1][0])     * inVec[i].val[3] ) + (2.0 + M + mesons.g*conj(sigma)) * inVec[i].val[0];
            outVec[i].val[1] += -im * mesons.g * ( conj((mesons.M[i+vol].val[0][0] - sigma)) * inVec[i].val[0] +         conj(mesons.M[i+vol].val[1][0])     * inVec[i].val[2] ) + (2.0 + M + mesons.g*conj(sigma)) * inVec[i].val[1];
            outVec[i].val[2] +=  im * mesons.g * (       conj(mesons.M[i+vol].val[0][1])     * inVec[i].val[1] + conj((mesons.M[i+vol].val[1][1] - sigma))   * inVec[i].val[3] ) + (2.0 + M + mesons.g*conj(sigma)) * inVec[i].val[2];
            outVec[i].val[3] += -im * mesons.g * (       conj(mesons.M[i+vol].val[0][1])     * inVec[i].val[0] + conj((mesons.M[i+vol].val[1][1] - sigma))   * inVec[i].val[2] ) + (2.0 + M + mesons.g*conj(sigma)) * inVec[i].val[3];
        } else{
            outVec[i].val[0] +=  im * mesons.g * ( (mesons.M[i+vol].val[0][0] - sigma) * inVec[i].val[1] +         mesons.M[i+vol].val[0][1]     * inVec[i].val[3] ) + (2.0 + M + mesons.g*sigma) * inVec[i].val[0];
            outVec[i].val[1] += -im * mesons.g * ( (mesons.M[i+vol].val[0][0] - sigma) * inVec[i].val[0] +         mesons.M[i+vol].val[0][1]     * inVec[i].val[2] ) + (2.0 + M + mesons.g*sigma) * inVec[i].val[1];
            outVec[i].val[2] +=  im * mesons.g * (       mesons.M[i+vol].val[1][0]     * inVec[i].val[1] + (mesons.M[i+vol].val[1][1] - sigma)   * inVec[i].val[3] ) + (2.0 + M + mesons.g*sigma) * inVec[i].val[2];
            outVec[i].val[3] += -im * mesons.g * (       mesons.M[i+vol].val[1][0]     * inVec[i].val[0] + (mesons.M[i+vol].val[1][1] - sigma)   * inVec[i].val[2] ) + (2.0 + M + mesons.g*sigma) * inVec[i].val[3];
        }

    }
}

template <typename Titer, typename Tvar>
void DiracOP<Titer, Tvar>::D_eo(Titer inVec,Titer outVec, MatrixType const useDagger){

    std::vector<int> idx(2);
    int const vol = l.vol/2;


    int const Nt=l.Nt, Nx=l.Nx;

    Tvar sgn[2];
    int nt, nx;
    
    for(int i=0; i<vol; i++){

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<Tvar> psisum[2], psidiff[2];
        
        if (useDagger == MatrixType::Dagger) {

            psisum[0]  = inVec[l.IDN[i][1] - vol].val[0] + inVec[l.IDN[i][1] - vol].val[1];
            psisum[1]  = inVec[l.IDN[i][1] - vol].val[2] + inVec[l.IDN[i][1] - vol].val[3];
            psidiff[0] = inVec[l.IUP[i][1] - vol].val[0] - inVec[l.IUP[i][1] - vol].val[1];
            psidiff[1] = inVec[l.IUP[i][1] - vol].val[2] - inVec[l.IUP[i][1] - vol].val[3];

            outVec[l.toEOflat(nt, nx)].val[0] -=  sgn[1] * inVec[l.IDN[i][0] - vol].val[0] + 0.5*psidiff[0] + 0.5*psisum[0];
            outVec[l.toEOflat(nt, nx)].val[2] -=  sgn[1] * inVec[l.IDN[i][0] - vol].val[2] + 0.5*psidiff[1] + 0.5*psisum[1];
            outVec[l.toEOflat(nt, nx)].val[1] -=  sgn[0] * inVec[l.IUP[i][0] - vol].val[1] - 0.5*psidiff[0] + 0.5*psisum[0];
            outVec[l.toEOflat(nt, nx)].val[3] -=  sgn[0] * inVec[l.IUP[i][0] - vol].val[3] - 0.5*psidiff[1] + 0.5*psisum[1];

        } else {

            psisum[0]  = inVec[l.IUP[i][1] - vol].val[0] + inVec[l.IUP[i][1] - vol].val[1];
            psisum[1]  = inVec[l.IUP[i][1] - vol].val[2] + inVec[l.IUP[i][1] - vol].val[3];
            psidiff[0] = inVec[l.IDN[i][1] - vol].val[0] - inVec[l.IDN[i][1] - vol].val[1];
            psidiff[1] = inVec[l.IDN[i][1] - vol].val[2] - inVec[l.IDN[i][1] - vol].val[3];

            outVec[l.toEOflat(nt, nx)].val[0] -=  sgn[0] * inVec[l.IUP[i][0] - vol].val[0] + 0.5*psisum[0] + 0.5*psidiff[0];
            outVec[l.toEOflat(nt, nx)].val[2] -=  sgn[0] * inVec[l.IUP[i][0] - vol].val[2] + 0.5*psisum[1] + 0.5*psidiff[1];
            outVec[l.toEOflat(nt, nx)].val[1] -=  sgn[1] * inVec[l.IDN[i][0] - vol].val[1] + 0.5*psisum[0] - 0.5*psidiff[0];
            outVec[l.toEOflat(nt, nx)].val[3] -=  sgn[1] * inVec[l.IDN[i][0] - vol].val[3] + 0.5*psisum[1] - 0.5*psidiff[1];

        }
                                            
    }

}

template <typename Titer, typename Tvar>
void DiracOP<Titer, Tvar>::D_oe(Titer inVec, Titer outVec, MatrixType const useDagger){
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Hopping term
    int const Nt=l.Nt, Nx=l.Nx;

    Tvar sgn[2];
    int nt, nx;

    int i;
    for(int j=0; j<vol; j++){

        i = j+vol; // full index (for lookup tables)

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<Tvar> psisum[2], psidiff[2];
        
        if (useDagger == MatrixType::Dagger) {

            psisum[0]  = inVec[l.IDN[i][1]].val[0] + inVec[l.IDN[i][1]].val[1];
            psisum[1]  = inVec[l.IDN[i][1]].val[2] + inVec[l.IDN[i][1]].val[3];
            psidiff[0] = inVec[l.IUP[i][1]].val[0] - inVec[l.IUP[i][1]].val[1];
            psidiff[1] = inVec[l.IUP[i][1]].val[2] - inVec[l.IUP[i][1]].val[3];

            outVec[j].val[0] -=  sgn[1] * inVec[l.IDN[i][0]].val[0] + 0.5*(psidiff[0] + psisum[0]);
            outVec[j].val[2] -=  sgn[1] * inVec[l.IDN[i][0]].val[2] + 0.5*(psidiff[1] + psisum[1]);
            outVec[j].val[1] -=  sgn[0] * inVec[l.IUP[i][0]].val[1] - 0.5*(psidiff[0] - psisum[0]);
            outVec[j].val[3] -=  sgn[0] * inVec[l.IUP[i][0]].val[3] - 0.5*(psidiff[1] - psisum[1]);

        } else {

            psisum[0]  = inVec[l.IUP[i][1]].val[0] + inVec[l.IUP[i][1]].val[1];
            psisum[1]  = inVec[l.IUP[i][1]].val[2] + inVec[l.IUP[i][1]].val[3];
            psidiff[0] = inVec[l.IDN[i][1]].val[0] - inVec[l.IDN[i][1]].val[1];
            psidiff[1] = inVec[l.IDN[i][1]].val[2] - inVec[l.IDN[i][1]].val[3];

            outVec[j].val[0] -=  sgn[0] * inVec[l.IUP[i][0]].val[0] + 0.5*psisum[0] + 0.5*psidiff[0];
            outVec[j].val[2] -=  sgn[0] * inVec[l.IUP[i][0]].val[2] + 0.5*psisum[1] + 0.5*psidiff[1];
            outVec[j].val[1] -=  sgn[1] * inVec[l.IDN[i][0]].val[1] + 0.5*psisum[0] - 0.5*psidiff[0];
            outVec[j].val[3] -=  sgn[1] * inVec[l.IDN[i][0]].val[3] + 0.5*psisum[1] - 0.5*psidiff[1];


        }
                                            
    }
}

template <typename Titer, typename Tvar>
void DiracOP<Titer, Tvar>::applyN(Titer inBegin, Titer inEnd, Titer outBegin){
    for(; inBegin != inEnd; inBegin++, outBegin++){
        outBegin->val[0] = mesons.g * (inBegin->val[0] - inBegin->val[1] + (im+1.0) * inBegin->val[3]);
        outBegin->val[1] = mesons.g * (inBegin->val[0] + inBegin->val[1] - (im+1.0) * inBegin->val[2]);
        outBegin->val[2] = mesons.g * ((im-1.0) * inBegin->val[1] + inBegin->val[2] + inBegin->val[3]);
        outBegin->val[3] = mesons.g * ((1.0-im) * inBegin->val[0] - inBegin->val[2] + inBegin->val[3]);
    }
}




