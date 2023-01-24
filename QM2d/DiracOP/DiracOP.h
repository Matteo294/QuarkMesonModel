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
        SpinorField applyToCSR(SpinorField const& inPsi, bool const dagger=0);
        O4Mesons* mesons;
        double const M;
        PauliMat Pauli;

    private:
        std::vector<mat> const GammaMat; // Wilson projectors and dagger
        // Diagonal term Dirac operator
        mat Diag {{2.0 + M, 0}, {0, 2.0 + M}};
};