#if !defined(USE_cuFFT) && !defined(USE_vkFFT)
#define USE_cUFFT
#endif

#include <cmath>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/complex.h>
#include <cooperative_groups/reduce.h>

#include "params.h"
#include "reductions.cuh"
#include "Spinor.cuh"
#include "Dirac.cuh"
#include "Lattice.cuh"

__global__ void Run(myType *eps, myType ExportTime, myType *field,
		myType *drift, myType *noise, int size, int *I, int *J, myType *vals, myType *maxDrift);
