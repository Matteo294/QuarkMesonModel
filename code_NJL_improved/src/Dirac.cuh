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
typedef cp = thrust::complex;

template <typename T>
class DiracOP {
	public:
		DiracOP();
		~DiracOP(){;}
		__host__  void applyD();
		__host__  void applyDhat(cp<double> *inVec, cp<double> *outVec);
		__host__ __device__ void setInVec(cp<T> *v){inVec = v;}
		__host__ __device__ void setOutVec(cp<T> *v){outVec = v;}
		__host__ __device__ void setDagger(MatrixType const Mtype){useDagger = Mtype;}
		__host__ __device__ void setM(cp<T> *mesonsMat){M = mesonsMat;}

		// Move into private:
		dim3 dimGrid_Dee, dimGrid_Doo, dimGrid_Doe, dimGrid_Deo;
		dim3 dimBlock_Dee, dimBlock_Doo, dimBlock_Doe, dimBlock_Deo;
		void *diagArgs[4]; // arguments for Dee, Doo, Deeinv, Dooinv
		void *hoppingArgs[5]; // arguments for Deo, Doe
		
	private:
		Spinor<T> temp, temp2;
		thrust::complex<T> *M;
		
		MatrixType useDagger;
		
		cp<T> *inVec, *outVec;
		LookUpTable IUP, IDN;

};

template <typename T>
__global__ void D_ee(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, cp<T> *M);
template <typename T>
__global__ void D_oo(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, cp<T> *M);
template <typename T>
__global__ void D_eo(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
template <typename T>
__global__ void D_oe(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
