#pragma once

#include <thrust/complex.h>
#include <random>
#include "Dirac.cuh"
#include "params.h"
#include "CGsolver.cuh"


class FermionicDrift{
    public:
        FermionicDrift();
        ~FermionicDrift(){cudaFree(afterCG); cudaFree(buf); cudaFree(vec); cudaFree(eobuf);};
		void getForce(double *outVec, DiracOP<double>& D, thrust::complex<double> *M, CGsolver& CG, dim3 dimGrid_drift, dim3 dimBlock_drift);
        //void setDirac(DiracOP<double> *Dirac){D = Dirac;}
    private:
        Spinor<double> *afterCG, *buf, *vec;
		thrust::complex<double> *eobuf;
		std::random_device rd; 
		std::mt19937 gen; 
		std::normal_distribution<float> dist;
		dim3 dimGrid_zero, dimBlock_zero;
		dim3 dimGrid_conv, dimBlock_conv;
        void *setZeroArgs[2];
		void *driftArgs[4];
		void *convArgs[2];
		int const spinor_vol = 4*vol;


};

__global__ void computeDrift(Spinor<double> *inVec, Spinor<double> *afterCG, thrust::complex<double> *outVec, int const vol);

__global__ void eoConv(thrust::complex<double> *eoVec, double *normalVec);
