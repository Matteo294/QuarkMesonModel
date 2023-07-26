#pragma once

#include <thrust/complex.h>
#include <random>
#include "Dirac.cuh"
#include "params.h"
#include "CGsolver.cuh"
#include <curand_kernel.h>


class FermionicDrift{
    public:
        FermionicDrift(int const seed);
        ~FermionicDrift(){cudaFree(eobuf); cudaFree(state);}
		void getForce(double *outVec, DiracOP<double>& D, CGsolver& CG, dim3 dimGrid_drift, dim3 dimBlock_drift);
        //void setDirac(DiracOP<double> *Dirac){D = Dirac;}
    private:
        Spinor<double> afterCG, buf, noiseVec;
		cp<double> *eobuf;
		std::random_device rd; 
		std::mt19937 gen; 
		std::normal_distribution<float> dist;
		dim3 dimGrid_zero, dimBlock_zero;
		dim3 dimGrid_conv, dimBlock_conv;
		dim3 dimGrid_rnd, dimBlock_rnd;
        void *setZeroArgs[2];
		void *driftArgs[4];
		void *convArgs[2];
		void *rndArgs[3];
		int const spinor_vol = 4*vol;
		curandState *state;
        LookUpTableConv N2EO;


};

__global__ void computeDrift(cp<double> *afterCG, cp<double> *noise, double *outVec, int *N2EO);

__global__ void eoConv(cp<double> *eoVec, double *normalVec);

__global__ void fillNormalRND(cp<double> *vec, curandState *dev_state, int const vol);

__global__ void random_setup_kernel(int const seed, curandState *state, int const vol);

