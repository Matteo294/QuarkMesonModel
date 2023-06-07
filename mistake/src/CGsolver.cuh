#pragma once

#include <cooperative_groups.h>
#include "reductions.cuh"
#include "Spinor.cuh"
#include "Lattice.cuh"
#include "Dirac.cuh"
#include "reductions.cuh"

__global__ void gpuDotProduct(thrust::complex<double> *vecA, thrust::complex<double> *vecB, thrust::complex<double> *result, int size);
__global__ void gpuSumSpinors(Spinor<double> *s1, Spinor<double> *s2, Spinor<double> *res, thrust::complex<double> c); //  = s1 + c * s2;

template <typename T>
class DiracOP;

class CGsolver{
    public:
        CGsolver();
        ~CGsolver();
        void solve(Spinor<double> *inVec, Spinor<double> *outVec, DiracOP<double>& D, MatrixType Mtype=MatrixType::Normal);
    private:
        thrust::complex<double> *dot_res;
        Spinor<double> *r, *p, *temp, *temp2, *sol;
        dim3 dimGrid_dot, dimBlock_dot;
        dim3 dimGrid_zero, dimBlock_zero;
        dim3 dimGrid_sum, dimBlock_sum;
        dim3 dimGrid_copy, dimBlock_copy;
        thrust::complex<double> beta, alpha;
		int const spinor_vol = 4 * vol;
};
