#pragma once
#include <vector>
#include <complex>
#include <random>
#include <cassert> 
#include "../functions/functions.h"
#include "../DiracOP/DiracOP.h"
#include "../Lattice/Lattice.h"




class Lattice;
typedef class SpinorField {
    public: 
        SpinorField(int const vol);
        SpinorField(SpinorField const& s);
        ~SpinorField() {;}
        void operator = (SpinorField const&);
        std::vector<Spinor> psi;
        void setZero(){for (auto& vec : psi){for (auto& val : vec.val) val = 0.0;}}
        //vecfield_iter oddBegin() {auto it = psi.begin(); std::advance(it, vol/2); return it;} // eventually return val.begin() + vol/2
    private:
        int const vol;
        std::random_device rnddev;
        std::mt19937 rndgen;
        std::normal_distribution<double> dist;
} SpinorField;

