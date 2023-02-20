#pragma once
#include "../SpinorField/SpinorField.h"
#include "../DiracOP/DiracOP.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../functions/functions.h"

using mat = Eigen::Matrix2cd; // Matrix in vecfield space OR flavour space (2x2)
using mat_fc = Eigen::Matrix<std::complex<double>, 4, 4>; // Matrix in flavour tensor vecfield space (composite matrix, 4x4)
using vec = Eigen::Vector2cd; // Vector in vecfield space OR flavour space (2 components)
using vec_fc = Eigen::Vector<std::complex<double>, 4>; // Vector in flavour tensor vecfield space (4 components)

class DiracOP;
struct O4Mat;
struct O4Mat_single;


class ConjugateGradientSolver{
    public:
        ConjugateGradientSolver(int const IterMax, double const tolerance, DiracOP& Dirac);
        ~ConjugateGradientSolver(){}
        void doubleCG_D(vecfield_iter ybegin, vecfield_iter yend, vecfield_iter xbegin);
        void doubleCG_Dhat(vecfield_iter ybegin, vecfield_iter yend, vecfield_iter xbegin);
        //void singleCG(vecfield_single_iter ybegin, vecfield_single_iter yend, vecfield_single_iter xbegin);
        //void mixedCG(vecfield_iter ybegin, vecfield_iter yend, vecfield_iter xbegin, int const IterMaxSingle, double const toleranceSingle);
    protected:
        double const tolerance;
        int const IterMax;
        DiracOP& Dirac;
};