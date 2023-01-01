#pragma once
#include <iostream>
#include <random>
#include <vector>
#include "../Mesons/O4Mesons.h"
#include "../functions/functions.h"
#include "../params.h"
#include <fftw3.h>

class O4Mesons;

class Langevin{
    public:
        Langevin(O4Mesons* mesons);
        ~Langevin();
        void LangevinRun(double dt, double T);
        int acceptance;
        O4Mesons* mesons;
    private:
        std::random_device rd_gaussian;
        std::mt19937 seed_gaussian;
        std::normal_distribution<> gaussian;
        std::random_device rd_uniform;
        std::mt19937 seed_uniform;
        std::uniform_real_distribution<double> uniform;

};
