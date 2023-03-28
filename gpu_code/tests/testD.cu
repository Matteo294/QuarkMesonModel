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

	thrust::complex<double> *M;
	Spinor<double> *in, *out;
	Lattice lattice(Nt, Nx);
	DiracOP<double> Dirac(fermion_mass, g_coupling, lattice);

	// Allocate two vectors and mesons matrix
	cudaMallocManaged(&M, sizeof(thrust::complex<double>) * 4 * lattice.vol);
	cudaMallocManaged(&in, sizeof(Spinor<double>) * lattice.vol);
	cudaMallocManaged(&out, sizeof(Spinor<double>) * lattice.vol);
	
	// Set fields values
	for(int i=0; i<lattice.vol; i++){
		M[i] = sigma + im * pi[2];
		M[i + 3*lattice.vol] = sigma - im * pi[2];
		M[i + lattice.vol] = im * (pi[0] - im * pi[1]);
		M[i + 2*lattice.vol] = im * (pi[0] + im * pi[1]);
	}
	for(int i=0; i<lattice.vol; i++){in[i].setZero();}
	for(int i=0; i<lattice.vol; i++){
		auto idx = lattice.eoToVec(i);
		in[i].val[0] = 1.0 * exp(im*idx[1]*q+im*idx[0]*p);
		in[i].val[1] = 1.0 * exp(im*idx[1]*q+im*idx[0]*p);
	}

	MatrixType useDagger = MatrixType::Normal;
	// diagArgs should be passed to all the diagonal (in spacetime) functions: Doo, Dee, Dooinv, Deeinv
	void *diagArgs[] = {(void*)&in, (void*)&out, (void*) &lattice.vol, (void*) &fermion_mass, (void*) &g_coupling, (void*)&useDagger, (void*)&M};
	// hopping should be passed to all the off-diagonal (in spacetime) functions: Deo, Doe
	void *hoppingArgs[] = {(void*)&in, (void*) &out, (void*) &lattice.vol, (void*) &useDagger, (void*) &lattice.IUP, (void*) &lattice.IDN};


	int numBlocks = 0;
	int numThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, gpuDotProduct);
	cudaDeviceSynchronize();

	useDagger = MatrixType::Normal;
	diagArgs[0] = (void*) &in; diagArgs[1] = (void*) &out;
	hoppingArgs[0] = (void*) &in; hoppingArgs[1] = (void*) &out;
	for(int i=0; i<lattice.vol; i++){out[i].setZero();}
	Dirac.applyD(diagArgs, hoppingArgs);
	cudaDeviceSynchronize();

	//std::ofstream myfile;
	//myfile.open("planewave.csv");
	//myfile << "nt,nx,v1,v2,v3,v4" << std::endl;
	int i;
	thrust::complex<double> v, w;
	cpdouble r = 0.0;
	for(int nx=0; nx<Nx; nx++){
		for(int nt=0; nt<Nt; nt++){
			i = lattice.toEOflat(nt, nx);
			v = (fermion_mass + g_coupling*sigma + 2.0*pow(sin(0.5*p), 2) + im * sin(p) + 2.0 * pow(sin(0.5*q), 2)) * in[i].val[0] + (-g_coupling*pi[2] + im*sin(q)) * in[i].val[1];
			w = (fermion_mass + g_coupling*sigma + 2.0*pow(sin(0.5*p), 2) - im * sin(p) + 2.0 * pow(sin(0.5*q), 2)) * in[i].val[1] + ( g_coupling*pi[2] + im*sin(q)) * in[i].val[0];
			/*myfile  	<< nt << "," << nx << "," << out[i].val[0].real() << "," << out[i].val[1].real() << ","
						<< v.real() << ","
						<< w.real() << "\n";*/
			std::cout 	<< "Site nt=" << nt << " nx=" << nx << "\n"  
						<< "1st component \t --> \t measured: " << out[i].val[0] << "\t\t expected :" << v << "\n"
						<< "2nd component \t --> \t measured: " << out[i].val[1] << "\t\t expected :" << w << "\n\n";
		}
	}


 
	std::cout << "Last error: " << cudaGetLastError() << ": " << cudaGetErrorString(cudaGetLastError()) << "\n";

	cudaFree(M);
	cudaFree(in);
	cudaFree(out);
	
	return 0;
}


__global__ void gpuDotProduct(cpdouble *vecA, cpdouble *vecB, cpdouble *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
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