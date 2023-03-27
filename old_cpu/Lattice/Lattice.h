#pragma once

#include <vector>

typedef class Lattice {
    public:
        Lattice(int const Nt, int const Nx);
        ~Lattice() {;}
        std::vector<std::vector<int>> IUP;
        std::vector<std::vector<int>> IDN;
        unsigned int PBC(int const n, int const N);
        unsigned int toEOflat(int const nt, int const nx);
        std::vector<int> eoToVec(int n);
        int const Nt, Nx, vol;
} Lattice;