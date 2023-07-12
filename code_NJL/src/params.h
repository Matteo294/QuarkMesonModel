#pragma once

#include <array>
#include <cstdio>
#include <cassert>
#include <thrust/complex.h>

enum class MatrixType {Normal, Dagger};

thrust::complex<double> const im {0.0, 1.0};

constexpr char CGmode = '0';
constexpr double tolerance = 1e-12;
constexpr int IterMax = 1000;

using myType = double;
//using myType = float;

int constexpr nDim = 2;
int constexpr nElements = (1+2*nDim);
int constexpr nVectorComponents = 1;

using dimArray = std::array<int, nDim>;

dimArray constexpr Sizes = {128, 128};
int const vol = Sizes[0]*Sizes[1];

template <int N>
int constexpr Prod() { return Sizes[N-1] * Prod<N-1>(); }

template <>
int constexpr Prod<0>() { return 1; }

int constexpr SIZE = Prod<nDim>();
int constexpr nTimeSlices = Sizes[0];
int constexpr SpatialVolume = SIZE / nTimeSlices;

int constexpr nFreq = ((Sizes[nDim-1]/2) + 1) * Prod<nDim-1>();

#define DEBUG

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}


