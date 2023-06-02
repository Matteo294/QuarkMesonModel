#pragma once

#include <cooperative_groups.h>
#include "reductions.cuh"
#include "Spinor.cuh"
#include "Lattice.cuh"
#include "Dirac.cuh"
#include "reductions.cuh"

__global__ void gpuDotProduct(thrust::complex<double> *vecA, thrust::complex<double> *vecB, thrust::complex<double> *result, int size);

class CGsolver{
    public:
        CGsolver();
        ~CGsolver();
        void solve(Spinor<double>  *inVec, Spinor<double> *outVec, DiracOP<double>& D, MatrixType MType=MatrixType::Normal);
    private:
        thrust::complex<double> *dot_res;
        Spinor<double> *r, *p, *temp, *temp2, *sol;
        dim3 dimGrid_dot, dimBlock_dot;
        dim3 dimGrid_setZero, dimBlock_setZero;

};