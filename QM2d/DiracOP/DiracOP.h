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


typedef struct PauliMat{
    mat const tau0 {{1, 0}, {0, 1}}; // identity
    mat const tau1 {{0, 1}, {1, 0}};
    mat const tau2 {{0, -im}, {im, 0}};
    mat const tau3 {{1, 0}, {0, -1}};
    std::vector<mat> tau {tau1, tau2, tau3};
} PauliMat;

class DiracOP {
    public:
        DiracOP(double const M, O4Mesons* mesons, Lattice& l);
        ~DiracOP(){;}

        void applyTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res, bool const dagger=0); // apply full Dirac operator to vector of dimension Nt*Nx
        void applyTo_single(std::vector<vec_fc_single>::iterator vec, std::vector<vec_fc_single>::iterator res, bool const dagger=0); // apply full Dirac operator to vector of dimension Nt*Nx


        //void applyTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res, bool const dagger=0);
        
        void applyDhatTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res, bool const dagger=0); // apply Dhat = Dee - Deo Doo_inv Doe to a vector of dimension Nt*Nx/2
        void applyDhatTo_single(std::vector<vec_fc_single>::iterator vec, std::vector<vec_fc_single>::iterator res, bool const dagger);

        void applyLTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res); // apply L to a vector of dimension Nt*Nx/2
        void applyRTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res); // apply R to a vector of dimension Nt*Nx/2
        void applyLinvTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res); // apply L_inv to a vector of dimension Nt*Nx/2
        void applyRinvTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res); // apply R_inv to a vector of dimension Nt*Nx/2

        // NB in the following functions the result is SUMMED to the vector passed !!!!!!!!!!!!!!!!!
        void D_ee(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger=0);
        void D_oo(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger=0);
        void D_oo_inv(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger=0);
        void D_eo(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger=0);
        void D_oe(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger=0);
        void D_ee_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger=0);
        void D_oo_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger=0);
        void D_oo_inv_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger=0);
        void D_eo_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger=0);
        void D_oe_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger=0);

        O4Mesons* mesons;
        double const M;
        PauliMat Pauli;
        Lattice& l;
};