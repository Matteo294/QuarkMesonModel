#pragma once

#include <thrust/complex.h>


template <typename T>
struct Spinor {
    public:
		Spinor(int vol){cudaMallocManaged(&val, sizeof(thrust::complex<double>) * 4*vol);}
		~Spinor(){cudaFree(val);}
		thrust::complex<double> data(){return val;}
	private:
    	thrust::complex<double> *val;
};
