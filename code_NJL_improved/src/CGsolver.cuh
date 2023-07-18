#pragma once

#include <cooperative_groups.h>
#include "reductions.cuh"
#include "Spinor.cuh"
#include "Lattice.cuh"
#include "Dirac.cuh"
#include "reductions.cuh"

using cp = thrust::complex;

__global__ void gpuDotProduct(cp<double> *vecA, cp<double> *vecB, cp<double> *result, int size);
__global__ void gpuSumSpinors(cp<double> *s1, cp<double> *s2, cp<double> *res, cp<double> c, int size); //  = s1 + c * s2;

template <typename T>
class DiracOP;

class CGsolver{
    public:
        CGsolver();
        ~CGsolver();
        void solve(cp<double> *inVec, cp<double> *outVec, DiracOP<double>& D, MatrixType Mtype=MatrixType::Normal);
        //void solveEO(cp<double> *inVec, cp<double> *outVec, DiracOP<double>& D, MatrixType Mtype=MatrixType::Normal);
    private:
        cp<double> *dot_res;
        Spinor<double> r, p, temp, temp2;
        dim3 dimGrid_dot, dimBlock_dot;
        dim3 dimGrid_zero, dimBlock_zero;
        dim3 dimGrid_sum, dimBlock_sum;
        dim3 dimGrid_copy, dimBlock_copy;
        void *dotArgs[4];
        void *setZeroArgs[2];
        void *sumArgs[5];
        void *copyArgs[3];
        cp<double> beta, alpha;
        int myvol;

};
