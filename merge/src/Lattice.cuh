#pragma once

#include <array>
#include <complex>
#include <iostream>
#include <thrust/complex.h>
#include <fstream>

#include "params.h"

struct my2dArray {
	int data[2];
	__host__ __device__ inline int& operator[](int i) {return data[i];}
	__host__ __device__ inline int operator[](int i) const {return data[i];}
};

struct LookUpTable{
    LookUpTable();
    ~LookUpTable();
    my2dArray *at;
};

__host__ __device__ unsigned int PBC(int const n, int const N);
__host__ __device__ unsigned int toEOflat(int const nt, int const nx);
__host__ __device__ my2dArray eoToVec(int n);
