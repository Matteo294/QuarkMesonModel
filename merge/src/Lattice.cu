#include "Lattice.cuh"
#include <array>
#include <complex>
#include <iostream>
#include <thrust/complex.h>
#include <fstream>

LookUpTable::LookUpTable(){cudaMallocManaged(&at, vol * sizeof(my2dArray));}
LookUpTable::~LookUpTable(){cudaFree(at);}



__host__ __device__ unsigned int PBC(int const n, int const N){
	return (n+N) % N;
}

__host__ __device__ unsigned int toEOflat(int const nt, int const nx){
	int const s = (Sizes[0]*Sizes[1])/2;
	int eo = (nt+nx) % 2;
	return (nx/2) + (nt*Sizes[1]/2) + eo*s;
}

__host__ __device__ my2dArray eoToVec(int n){
	my2dArray idx; // nt, nx
	int alpha = 0;
	if (n >= Sizes[0]*Sizes[1]/2) {
		alpha = 1;
		n -= Sizes[0]*Sizes[1]/2;
	}
	idx[0] = n / (Sizes[1]/2);
	if (idx[0] % 2) idx[1] = 2*((n % (Sizes[1]/2))) + (1-alpha);
	else idx[1] = 2*((n % (Sizes[1]/2))) + alpha; 
	return idx;
}

__host__ __device__ int EOtoNormal(int n){
	my2dArray idx = eoToVec(n);
	return idx[1] + Sizes[1]*idx[0];
}

__host__ __device__  int NormalToEO(int n){
	int nt = n / Sizes[1];
	int nx = n % Sizes[1];
	return toEOflat(nt, nx);
}
