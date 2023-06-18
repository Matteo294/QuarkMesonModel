#pragma once

#include <array>
#include <complex>
#include <iostream>
#include "Spinor.cuh"
#include "Lattice.cuh"
#include "CGsolver.cuh"
#include <cooperative_groups.h>
#include "params.h"

class CGsolver;

namespace cg = cooperative_groups;


template <typename T>
class DiracOP {
	public:
		DiracOP();
		~DiracOP(){cudaFree(temp); cudaFree(temp2);}
		__host__  void applyD();
		__host__ __device__ void setInVec(Spinor<T> *v){inVec = v;}
		__host__ __device__ void setOutVec(Spinor<T> *v){outVec = v;}
		__host__ __device__ void setDagger(MatrixType const Mtype){useDagger = Mtype;}
		__host__ __device__ void setM(thrust::complex<T> *mesonsMat){M = mesonsMat;}
	private:
		Spinor<T> *temp, *temp2;
		thrust::complex<T> *M;
		dim3 dimGrid_Dee, dimGrid_Doo, dimGrid_Doe, dimGrid_Deo;
		dim3 dimBlock_Dee, dimBlock_Doo, dimBlock_Doe, dimBlock_Deo;
		MatrixType useDagger;
		void *diagArgs[4]; // arguments for Dee, Doo, Deeinv, Dooinv
		void *hoppingArgs[5]; // arguments for Deo, Doe
		Spinor<T> *inVec, *outVec;
		LookUpTable IUP, IDN;

};

template <typename T>
__global__ void D_oo_inv(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, thrust::complex<T> *M);
template <typename T>
__global__ void D_ee_inv(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, thrust::complex<T> *M);
template <typename T>
__global__ void D_ee(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, thrust::complex<T> *M);
template <typename T>
__global__ void D_oo(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, thrust::complex<T> *M);
template <typename T>
__global__ void D_eo(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
template <typename T>
__global__ void D_oe(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);