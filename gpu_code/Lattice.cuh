#pragma once

#include <array>
#include <complex>
#include <iostream>
#include <thrust/complex.h>
#include <fstream>

struct my2dArray {
	int data[2];
	__host__ __device__ inline int& operator[](int i) {return data[i];}
	__host__ __device__ inline int operator[](int i) const {return data[i];}
};

class Lattice {
    public:
        __host__ ~Lattice() {cudaFree(IUP); cudaFree(IDN);}
        __host__ Lattice(int const Nt, int const Nx);
        my2dArray *IUP, *IDN;
        __host__ __device__ unsigned int PBC(int const n, int const N);
        __host__ __device__ unsigned int toEOflat(int const nt, int const nx);
        __host__ __device__ my2dArray eoToVec(int n);
        int const Nt, Nx, vol;
};
