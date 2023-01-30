#pragma once
#include "../SpinorField/SpinorField.h"
#include "../DiracOP/DiracOP.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../functions/functions.h"

using mat = Eigen::Matrix2cd; // Matrix in spinor space OR flavour space (2x2)
using mat_fc = Eigen::Matrix<std::complex<double>, 4, 4>; // Matrix in flavour tensor spinor space (composite matrix, 4x4)
using vec = Eigen::Vector2cd; // Vector in spinor space OR flavour space (2 components)
using vec_fc = Eigen::Vector<std::complex<double>, 4>; // Vector in flavour tensor spinor space (4 components)

class DiracOP;

class ConjugateGradientSolver{
    public:
        ConjugateGradientSolver(int const IterMax, double const tolerance, DiracOP& Dirac);
        ~ConjugateGradientSolver(){;}
        void baseCG(SpinorField const& inPsi, SpinorField& outPsi);
        void eoCG(SpinorField const& inPsi, SpinorField& outPsi);
        //void MixedCG(SpinorField const& inPsi, SpinorField& outPsi);
        //void eoMixedCG(SpinorField const& inPsi, SpinorField& outPsi);
    protected:
        double const tolerance;
        int const IterMax;
        DiracOP& Dirac;
};