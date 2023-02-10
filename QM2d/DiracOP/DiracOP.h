#pragma once

#include <vector>
#include <complex>
#include "../SpinorField/SpinorField.h"
#include "../Mesons/O4Mesons.h"
#include "../functions/functions.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

class SpinorField;
class O4Mesons;
class Lattice;

class DiracOP {
    public:
        DiracOP(double const M, O4Mesons& mesons, Lattice& l);
        ~DiracOP(){}

        template <typename T>        
        void applyTo(T vec, T res, MatrixType const useDagger=MatrixType::Normal); // apply full Dirac operator to vector of dimension Nt*Nx
        void applyTo_single(spinor_single_iter vec, spinor_single_iter res, MatrixType const useDagger=MatrixType::Normal); // apply full Dirac operator to vector of dimension Nt*Nx
        void applyDhatTo(spinor_iter vec, spinor_iter res, MatrixType const useDagger=MatrixType::Normal); // apply Dhat = Dee - Deo Doo_inv Doe to a vector of dimension Nt*Nx/2

        // NB in the following functions the result is ASSIGNED to the vector passed !!!!!!!!!!!!!!!!!
        void D_oo_inv(spinor_iter y, spinor_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_ee_inv(spinor_iter y, spinor_iter x, MatrixType const useDagger=MatrixType::Normal);

        // NB in the following functions the result is SUMMED to the vector passed !!!!!!!!!!!!!!!!!
        void D_ee(spinor_iter y, spinor_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_oo(spinor_iter y, spinor_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_eo(spinor_iter y, spinor_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_oe(spinor_iter y, spinor_iter x, MatrixType const useDagger=MatrixType::Normal);
        
        
        /*
        void D_ee_single(spinor_single_iter y, spinor_single_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_oo_single(spinor_single_iter y, spinor_single_iter x, MatrixType const useDagger=MatrixType::Normal);
        //void D_oo_inv_single(spinor_single_iter y, spinor_single_iter x, MatrixType const useDagger=MatrixType::Normal=0);
        void D_eo_single(spinor_single_iter y, spinor_single_iter x, MatrixType const useDagger=MatrixType::Normal);
        void D_oe_single(spinor_single_iter y, spinor_single_iter x, MatrixType const useDagger=MatrixType::Normal);
        void applyDhatTo_single(spinor_single_iter vec, spinor_single_iter res, MatrixType const useDagger=MatrixType::Normal);
        */

    typedef struct PauliMat{
        mat const tau0 {{1, 0}, {0, 1}}; // identity
        mat const tau1 {{0, 1}, {1, 0}};
        mat const tau2 {{0, -im}, {im, 0}};
        mat const tau3 {{1, 0}, {0, -1}};
        std::vector<mat> tau {tau1, tau2, tau3};
    } PauliMat;

    PauliMat Pauli;
    Lattice& l;
    O4Mesons& mesons;
    double const M;

};
