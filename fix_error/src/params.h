#pragma once

#include <array>
#include <cstdio>
#include <cassert>

enum class MatrixType {Normal, Dagger};

double constexpr yukawa_coupling = 1.0;
double constexpr fermion_mass = 0.2;

double constexpr tolerance = 1e-11;
int constexpr IterMax = 1000;
using myType = double;
//using myType = float;


int constexpr nDim = 2;
int constexpr nElements = (1+2*nDim);
int constexpr nVectorComponents = 4;

using dimArray = std::array<int, nDim>;
//dimArray constexpr Sizes = {8, 8};//, 12, 12};
//dimArray constexpr Sizes = {16, 16};//, 12, 12};
//dimArray constexpr Sizes = {12, 24};//, 12, 12};
//dimArray constexpr Sizes = {64, 64};//, 12, 12};
//dimArray constexpr Sizes = {128, 128};//, 12, 12};
//dimArray constexpr Sizes = {256, 256};//, 12, 12};

dimArray constexpr Sizes = {16, 16};
int const vol = Sizes[0]*Sizes[1];

/*
int constexpr nDim = 4;
int constexpr nElements = (1+2*nDim);
int constexpr nVectorComponents = 4;

using dimArray = std::array<int, nDim>;
dimArray constexpr Sizes = {12, 4, 4, 4};
*/

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


