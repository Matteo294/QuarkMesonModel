#include "Lattice.cuh"
#include <array>
#include <complex>
#include <iostream>
#include <thrust/complex.h>
#include <fstream>

__host__ Lattice::Lattice(int const Nt, int const Nx) :
	Nt{Nt},
	Nx{Nx},
	vol{Nt*Nx}
	{   
		//IUP = (my2dArray *) malloc(sizeof(my2dArray) * vol);
		//IDN = (my2dArray *) malloc(sizeof(my2dArray) * vol);
		cudaMallocManaged(&IUP, sizeof(my2dArray) * vol);
		cudaMallocManaged(&IDN, sizeof(my2dArray) * vol);
		my2dArray idx;
		for(int i=0; i<vol; i++){
			idx = eoToVec(i);
			IUP[i][0] = toEOflat(PBC(idx[0]+1, Nt), idx[1]);
			IUP[i][1] = toEOflat(idx[0], PBC(idx[1]+1, Nx));
			IDN[i][0] = toEOflat(PBC(idx[0]-1, Nt), idx[1]);
			IDN[i][1] = toEOflat(idx[0], PBC(idx[1]-1, Nx));
		}
	}

__host__ __device__ unsigned int Lattice::PBC(int const n, int const N){
	return (n+N) % N;
}

__host__ __device__ unsigned int Lattice::toEOflat(int const nt, int const nx){
	int const s = (Nt*Nx)/2;
	int eo = (nt+nx) % 2;
	return (nx/2) + (nt*Nx/2) + eo*s;
}

__host__ __device__ my2dArray Lattice::eoToVec(int n){
	my2dArray idx; // nt, nx
	int alpha = 0;
	if (n >= Nt*Nx/2) {
		alpha = 1;
		n -= Nt*Nx/2;
	}
	idx[0] = n / (Nx/2);
	if (idx[0] % 2) idx[1] = 2*((n % (Nx/2))) + (1-alpha);
	else idx[1] = 2*((n % (Nx/2))) + alpha; 
	return idx;
}
