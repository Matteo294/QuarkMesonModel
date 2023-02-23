#pragma once
#include "../SpinorField/SpinorField.h"
#include "../DiracOP/DiracOP.h"
#include "../functions/functions.h"
#include <algorithm>


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