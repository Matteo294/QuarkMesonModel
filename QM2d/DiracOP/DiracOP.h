#pragma once

#include <vector>
#include <complex>
#include "../SpinorField/SpinorField.h"
#include "../Mesons/O4Mesons.h"
#include "../functions/functions.h"


typedef class DiracOP {
    public:
        DiracOP(double const M, O4Mesons& mesons, Lattice& l);
        ~DiracOP(){}

        template <typename T>        
        void applyTo(T vec, T res, MatrixType const useDagger=MatrixType::Normal); // apply full Dirac operator to vector of dimension Nt*Nx
        void applyTo_single(vecfield_single_iter vec, vecfield_single_iter res, MatrixType const useDagger=MatrixType::Normal); // apply full Dirac operator to vector of dimension Nt*Nx
        void applyDhatTo(vecfield_iter vec, vecfield_iter res, MatrixType const useDagger=MatrixType::Normal); // apply Dhat = Dee - Deo Doo_inv Doe to a vector of dimension Nt*Nx/2

        // NB in the following functions the result is ASSIGNED to the vector passed !!!!!!!!!!!!!!!!!
        void D_oo_inv(vecfield_iter y, vecfield_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_ee_inv(vecfield_iter y, vecfield_iter x, MatrixType const useDagger=MatrixType::Normal);

        // NB in the following functions the result is SUMMED to the vector passed !!!!!!!!!!!!!!!!!
        void D_ee(vecfield_iter y, vecfield_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_oo(vecfield_iter y, vecfield_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_eo(vecfield_iter y, vecfield_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_oe(vecfield_iter y, vecfield_iter x, MatrixType const useDagger=MatrixType::Normal);

        void applyN(vecfield_iter inBegin, vecfield_iter inEnd, vecfield_iter outBegin);
        
    private:
        Lattice& l;
        O4Mesons& mesons;
        double const M;

} DiracOP;
