#pragma once

#include <cmath>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/complex.h>

#include "params.h"

__device__ void gpuMaxAbsReduce(myType *vecA, myType *result, int size);
__global__ void gpuTimeSlices(myType *vecA, myType *result, int size);
__global__ void gpuMagnetisation(myType *vecA, myType *result, int size);


// Matteo
__global__ void setZeroGPU(thrust::complex<double> *v, int const vol);