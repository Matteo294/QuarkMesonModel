#pragma once

#include <vector>
#include <complex>
#include "../SpinorField/SpinorField.h"
#include "../params.h"

class SpinorField;
typedef std::vector<std::vector<std::complex<double>>> mat;

class DiracOP {
    public:
        DiracOP(double const M);
        SpinorField applyTo(SpinorField const& inPsi, bool dagger=0);
        double const M;
    private:
        // Hopping terms Dirac operator
        mat const Gamma_p0 {{2, 0}, {0, 0}};
        mat const Gamma_m0 {{0, 0}, {0, 2}};
        mat const Gamma_p1 {{1, 1}, {1, 1}}; 
        mat const Gamma_m1 {{1, -1}, {-1, 1}};
        // Diagonal term Dirac operator
        mat Diag {{2.0 + M, 0}, {0, 2.0 + M}};

};