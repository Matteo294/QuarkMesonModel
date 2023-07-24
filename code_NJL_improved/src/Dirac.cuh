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
		~DiracOP(){;}

		__host__ __device__ void setInVec(cp<T> *v){inVec = v;}
		__host__ __device__ void setOutVec(cp<T> *v){outVec = v;}
		__host__ __device__ void setDagger(MatrixType const Mtype){useDagger = Mtype;}
		__host__ __device__ void setM(T *mesonsMat){M = mesonsMat;}

		// Move into private:
		dim3 dimGrid_Dee, dimGrid_Doo, dimGrid_Doe, dimGrid_Deo;
		dim3 dimBlock_Dee, dimBlock_Doo, dimBlock_Doe, dimBlock_Deo;
		void *diagArgs[5]; // arguments for Dee, Doo, Deeinv, Dooinv
		void *hoppingArgs[5]; // arguments for Deo, Doe
		
	
		Spinor<T> temp, temp2;
		T *M;
		
		MatrixType useDagger;
		
		cp<T> *inVec, *outVec;
		LookUpTable IUP, IDN;
        LookUpTableConv EO2N;


};

/*template <typename T>
__device__ void D_ee(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, T *M, int *EO2I);
template <typename T>
__device__ void D_oo(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, T *M, int *EO2I);
template <typename T>
__device__ void D_eo(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
template <typename T>
__device__ void D_oe(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);*/


__device__ void D_ee(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M, int *EO2N);
__device__ void D_oo(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M, int *EO2N);
__device__ void D_eo(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
__device__ void D_oe(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
