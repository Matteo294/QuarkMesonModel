#pragma once

#include <vector>
#include <complex>
#include "../SpinorField/SpinorField.h"
#include "../params.h"
#include "../Mesons/O4Mesons.h"
#include "../functions/functions.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

class SpinorField;
class O4Mesons;

using mat = Eigen::Matrix2cd;

typedef struct PauliMat{
    mat const tau0 {{1, 0}, {0, 1}}; // identity
    mat const tau1 {{0, 1}, {1, 0}};
    mat const tau2 {{0, -im}, {im, 0}};
    mat const tau3 {{1, 0}, {0, -1}};
    std::vector<mat> tau {tau1, tau2, tau3};
} PauliMat;

class DiracOP {
    public:
        DiracOP(double const M, O4Mesons* mesons);
        ~DiracOP(){;}
        SpinorField applyTo(SpinorField const& inPsi, bool dagger=0);
        void D_ee(SpinorField const& inPsi, SpinorField& outPsi);
        void D_oo(SpinorField const& inPsi, SpinorField& outPsi);
        void D_eo(SpinorField const& inPsi, SpinorField& outPsi, bool dagger=0);
        void D_oe(SpinorField const& inPsi, SpinorField& outPsi, bool dagger=0);
        O4Mesons* mesons;
        double const M;
        PauliMat Pauli;

    private:
        // Hopping terms Dirac operator
        mat const Gamma_p0, Gamma_m0, Gamma_p1, Gamma_m1;
        // Hopping terms Dirac dagger operator
        mat const Gamma_p0_dagger, Gamma_m0_dagger, Gamma_p1_dagger, Gamma_m1_dagger;
        // Diagonal term Dirac operator
        mat Diag {{2.0 + M, 0}, {0, 2.0 + M}};
};