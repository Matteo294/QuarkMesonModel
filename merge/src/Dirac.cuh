#pragma once

#include <array>
#include <complex>
#include <iostream>
#include <thrust/complex.h>
#include "Spinor.cuh"
#include "Lattice.cuh"
#include <cooperative_groups.h>
#include "params.h"


namespace cg = cooperative_groups;


template <typename T>
class DiracOP {
	public:
		__host__ DiracOP(T const f_mass, T const g, Lattice& l) : fermion_mass{f_mass}, g_coupling{g}, lattice{l}
		{cudaMallocManaged(&temp, sizeof(Spinor<T>) * lattice.vol/2); cudaMallocManaged(&temp2, sizeof(Spinor<T>) * lattice.vol/2);}
		__host__ ~DiracOP(){cudaFree(temp); cudaFree(temp2);}
		__host__ void applyD(void** diagArgs, void** hoppingArgs);
		
		//__host__ void applyDhat(Spinor<T> *inVec, Spinor<T> *outVec, MesonsMat<T> *M, MatrixType const useDagger);	   

		Lattice& lattice;
		T const fermion_mass;
	private:
		Spinor<T> *temp, *temp2;
		T const g_coupling;

};

template <typename T>
__global__ void D_oo_inv(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, T const fermion_mass, T const g_coupling, MatrixType const useDagger, thrust::complex<T> *M);
template <typename T>
__global__ void D_ee_inv(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, T const fermion_mass, T const g_coupling, MatrixType const useDagger, thrust::complex<T> *M);
template <typename T>
__global__ void D_ee(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, T const fermion_mass, T const g_coupling, MatrixType const useDagger, thrust::complex<T> *M);
template <typename T>
__global__ void D_oo(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, T const fermion_mass, T const g_coupling, MatrixType const useDagger, thrust::complex<T> *M);
template <typename T>
__global__ void D_eo(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
template <typename T>
__global__ void D_oe(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
