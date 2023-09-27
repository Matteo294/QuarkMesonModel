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
		void setScalar(T* phi){M = phi;}
		void applyD(cp<double> *in, cp<double> *out, MatrixType MType);
		T *M;
		LookUpTable IUP, IDN;
        LookUpTableConv EO2N;


};


__global__ void applyD_gpu(cp<double> *in, cp<double> *out, MatrixType const useDagger, double *M, int *EO2N, my2dArray *IDN, my2dArray *IUP);
__device__ void D_ee(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M, int *EO2N);
__device__ void D_oo(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M, int *EO2N);
__device__ void D_eo(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
__device__ void D_oe(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN);
