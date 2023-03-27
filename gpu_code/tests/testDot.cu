#include <array>
#include <complex>
#include <iostream>
#include <thrust/complex.h>
#include <fstream>
#include <cooperative_groups.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "../Dirac.cuh"
#include "../Spinor.cuh"
#include "../Lattice.cuh"
#include "../params.h"

namespace cg = cooperative_groups;

using std::conj;

double const q = 6.0 * (double) M_PI / Nx;
double const p = 7.0 * (double) M_PI / Nt;

thrust::complex<double> im {0.0, 1.0};

using cpdouble = thrust::complex<double>;

__global__ void gpuDotProduct(cpdouble *vecA, cpdouble *vecB, cpdouble *result, int size);

		
int main() {

	thrust::complex<double> *M1, *M2, *M3, *M4;
	Spinor<double> *in, *out;
	Lattice lattice(Nt, Nx);
	DiracOP<double> Dirac(fermion_mass, g_coupling, lattice);

	// Allocate two vectors and mesons matrix
	cudaMallocManaged(&M1, sizeof(thrust::complex<double>) * lattice.vol);
	cudaMallocManaged(&M2, sizeof(thrust::complex<double>) * lattice.vol);
	cudaMallocManaged(&M3, sizeof(thrust::complex<double>) * lattice.vol);
	cudaMallocManaged(&M4, sizeof(thrust::complex<double>) * lattice.vol);
	cudaMallocManaged(&in, sizeof(Spinor<double>) * lattice.vol);
	cudaMallocManaged(&out, sizeof(Spinor<double>) * lattice.vol);

    cpdouble *v, *res;
    cudaMallocManaged(&v, sizeof(cpdouble) * 128);
    cudaMallocManaged(&res, sizeof(cpdouble));
    
    v[0] = im;
	v[1] = 2.0;
    v[31] = 1;
	v[32] = -im;
	v[63] = 2*im;

    int numBlocks = 0;
	int numThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, gpuDotProduct);
	cudaDeviceSynchronize();
	std::cout << numBlocks << '\t' << numThreads << '\n';
    auto dimGrid = dim3(numBlocks, 1, 1);
	auto dimBlock = dim3(numThreads, 1, 1);
	int const mySize = (2<<7);
	void *dotArgs[] = {(void*) &v, (void*) &v, (void*) &res, (void*) &mySize};

	cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * (2<<7), NULL);
	cudaDeviceSynchronize();

	std::cout << *res << std::endl;

	std::cout << "Last error: " << cudaGetErrorString(cudaGetLastError()) << "\n";

	cudaFree(M1);
	cudaFree(M2);
	cudaFree(M3);
	cudaFree(M4);
	cudaFree(in);
	cudaFree(out);

    cudaFree(v);
    cudaFree(res);
	
	return 0;
}


__global__ void gpuDotProduct(cpdouble *vecA, cpdouble *vecB, cpdouble *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	int n = grid.thread_rank();
	*result = 0.0;
	extern __shared__ cpdouble tmp[];

	cpdouble temp_sum = 0.0;
	for (int i = grid.thread_rank(); i < size; i += grid.size()) {
		temp_sum += conj(vecA[i]) * vecB[i];
	}

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	temp_sum = cg::reduce(tile32, temp_sum, cg::plus<cpdouble>());

	if (tile32.thread_rank() == 0) {
		tmp[tile32.meta_group_rank()] = temp_sum;
	}

	cg::sync(cta);

	if (tile32.meta_group_rank() == 0) {
		temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
		temp_sum = cg::reduce(tile32, temp_sum, cg::plus<cpdouble>());

		if (tile32.thread_rank() == 0) {
		atomicAdd(reinterpret_cast<double*>(result), temp_sum.real());
		atomicAdd(reinterpret_cast<double*>(result)+1, temp_sum.imag());
		}
	}
}