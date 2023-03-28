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

double const q = 4.0 * (double) M_PI / Nx; // 2n
double const p = 7.0 * (double) M_PI / Nt; // 2n+1

thrust::complex<double> im {0.0, 1.0};

using cpdouble = thrust::complex<double>;

template <typename T>
__host__ void CGsolver_solve_D(Spinor<T> *inVec, Spinor<T> *outVec, DiracOP<T>& D, thrust::complex<double> *M, int const numBlocks, int const numThreads);

__global__ void gpuDotProduct(cpdouble *vecA, cpdouble *vecB, cpdouble *result, int size);

void getForce(Spinor<double> *inVec, cpdouble *outVec, DiracOP<double>& D, cpdouble *M, Lattice& lattice, int const nBlocks_dot, int const nThreads_dot, int const nBlocks_drift, int const nThreads_drift);
__global__ void computeDrift(Spinor<double> *inVec, Spinor<double> *afterCG, cpdouble *outVec, int const vol);

		
int main() {

	thrust::complex<double> *M;
	Spinor<double> *in, *v2;
	thrust::complex<double> *out;
	Lattice lattice(Nt, Nx);
	DiracOP<double> Dirac(fermion_mass, g_coupling, lattice);

	// Allocate two vectors and mesons matrix
	cudaMallocManaged(&M, sizeof(thrust::complex<double>) * 4 * lattice.vol);
	cudaMallocManaged(&in, sizeof(Spinor<double>) * lattice.vol);
	cudaMallocManaged(&v2, sizeof(Spinor<double>) * lattice.vol);
	cudaMallocManaged(&out, sizeof(cpdouble) * 4 * lattice.vol);
	
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

	// Calculate Occupancy
	int nBlocks_dot = 0, nBlocks_drift = 0;
	int nThreads_dot = 0, nThreads_drift = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks_dot, &nThreads_dot, gpuDotProduct);
	cudaDeviceSynchronize();
	cudaOccupancyMaxPotentialBlockSize(&nBlocks_drift, &nThreads_drift, computeDrift);
	cudaDeviceSynchronize();

	// Compute force
	getForce(in, out, Dirac, M, lattice, nBlocks_dot, nThreads_dot, nBlocks_drift, nThreads_drift);

	// Check results
	int i;
	int const vol = lattice.vol;
	thrust::complex<double> v, w = 0.0;
	double pi_sq = pi[0]*pi[0] + pi[1]*pi[1] + pi[2]*pi[2];
	for(int nx=0; nx<Nx; nx++){
		for(int nt=0; nt<Nt; nt++){
			i = lattice.toEOflat(nt, nx);
			v = 		g_coupling * ((fermion_mass + g_coupling*sigma + 2.0*sin(0.5*p)*sin(0.5*p) + 2.0*sin(0.5*q)*sin(0.5*q) - im*sin(p))  +
						(g_coupling*pi[2] - im*sin(q))) /
						(pow(fermion_mass + g_coupling*sigma + 2.0*sin(0.5*p)*sin(0.5*p) + 2.0*sin(0.5*q)*sin(0.5*q), 2) + sin(q)*sin(q) + sin(p)*sin(p) + g_coupling*g_coupling*pi_sq);
			w = 		g_coupling * ((fermion_mass + g_coupling*sigma + 2.0*sin(0.5*p)*sin(0.5*p) + 2.0*sin(0.5*q)*sin(0.5*q) + im*sin(p)) -
						(g_coupling*pi[2] + im*sin(q))) /
						(pow(fermion_mass + g_coupling*sigma + 2.0*sin(0.5*p)*sin(0.5*p) + 2.0*sin(0.5*q)*sin(0.5*q), 2) + sin(q)*sin(q) + sin(p)*sin(p) + g_coupling*g_coupling*pi_sq);
			std::cout 	<< "Site nt=" << nt << " nx=" << nx << "\n";
			std::cout 	<< "Drift for sigma --> " << "expected: " << conj(v) + conj(w) << " \t measured: " << out[i] << "\n";
			std::cout 	<< "dM: " 
						<< out[i] + im*out[i + 3*vol] << " \t " 
						<<  im * (out[i + vol] - im*out[i + 2*vol]) << " \t "
						<< -im * (out[i + vol] + im*out[i + 2*vol]) << " \t "
						<< out[i] - im*out[i + 3*vol] << " \n\n ";

		}
	}

	auto err = cudaGetLastError();
	std::cout << "Last error: " << err << " --> " << cudaGetErrorString(err) << "\n";

	cudaFree(M);
	cudaFree(in);
	cudaFree(out);
	cudaFree(v2);
	
	return 0;
}


template <typename T>
__host__ void CGsolver_solve_D(Spinor<T> *inVec, Spinor<T> *outVec, DiracOP<T>& D, thrust::complex<double> *M, int const numBlocks_dot, int const numThreads_dot){	
	
	int const vol = D.lattice.vol;
	int mySize = D.lattice.vol * 4;

	Spinor<T> *r, *p, *temp, *temp2; // allocate space ?? 
	thrust::complex<T> alpha; // allocate space ??
	T beta, rmodsq;
	cpdouble *dot_res;

	// Set up dot product call
	void *dotArgs[] = {(void*) &r, (void*) &r, (void*) &dot_res, (void*) &mySize};
	auto dimGrid = dim3(numBlocks_dot, 1, 1);
	auto dimBlock = dim3(numThreads_dot, 1, 1);

	cudaMallocManaged(&r, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&p, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&temp, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&temp2, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&dot_res, sizeof(cpdouble));

	for(int i=0; i<vol; i++) {
		outVec[i] = Spinor<T> ();
		temp[i] = Spinor<T> ();
		temp2[i] = Spinor<T> ();
		for(int j=0; j<4; j++) r[i].val[j] = inVec[i].val[j];
		for(int j=0; j<4; j++) p[i].val[j] = r[i].val[j];
	}

	cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * 32, NULL);
	cudaDeviceSynchronize();
	rmodsq = dot_res->real();

	MatrixType dag = MatrixType::Normal;

	void *diagArgs[] = {(void*)&p, (void*)&temp2, (void*) &D.lattice.vol, (void*) &fermion_mass, (void*) &g_coupling, (void*)&dag, (void*)&M};
	void *hoppingArgs[] = {(void*)&p, (void*)&temp2, (void*) &D.lattice.vol, (void*)&dag, (void*)&D.lattice.IUP, (void*)&D.lattice.IDN};

	int k;
	for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) temp[i].val[j] = 2.0 * p[i].val[j];
		}

		// Set buffers to zero to store the result fo the Dirac operator applied to p
		for(int i=0; i<D.lattice.vol; i++) {temp2[i].setZero(); temp[i].setZero();}

		// Apply D dagger
		dag = MatrixType::Dagger;
		diagArgs[0] = (void*) &p; diagArgs[1] = (void*) &temp2;
		hoppingArgs[0] = (void*) &p; hoppingArgs[1] = (void*) &temp2;
		D.applyD(diagArgs, hoppingArgs);
		// Apply D
		dag = MatrixType::Normal;
		diagArgs[0] = (void*) &temp2; diagArgs[1] = (void*) &temp;
		hoppingArgs[0] = (void*) &temp2; hoppingArgs[1] = (void*) &temp;
		D.applyD(diagArgs, hoppingArgs);
		
		dotArgs[0] = (void*) &p; dotArgs[1] = (void*) &temp;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * 32, NULL);
		cudaDeviceSynchronize();
		alpha = rmodsq / *dot_res; 

		// x = x + alpha p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) outVec[i].val[j] += alpha*p[i].val[j];
		}
		// r = r - alpha A p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) r[i].val[j] -= alpha*temp[i].val[j];
		}

		dotArgs[0] = (void*) &r; dotArgs[1] = (void*) &r;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * 32, NULL);
		cudaDeviceSynchronize();
		beta = dot_res->real() / rmodsq;

		// p = r - beta p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) p[i].val[j] = r[i].val[j] + beta*p[i].val[j];
		}

		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * 32, NULL);
		cudaDeviceSynchronize();
		rmodsq = dot_res->real();
	}

	if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
	else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << sqrt(rmodsq) << "\n";

	cudaFree(r);
	cudaFree(p);
	cudaFree(temp);
	cudaFree(temp2);
	cudaFree(dot_res);
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

void getForce(Spinor<double> *inVec, cpdouble *outVec, DiracOP<double>& D, cpdouble *M, Lattice& lattice, int const nBlocks_dot, int const nThreads_dot, int const nBlocks_drift, int const nThreads_drift){
	
	int const vol = lattice.vol;
	Spinor<double> *afterCG, *buf;
	cudaMallocManaged(&afterCG, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&buf, sizeof(Spinor<double>) * vol);

	for(int i=0; i<vol; i++){ afterCG[i].setZero(); buf[i].setZero();}

	CGsolver_solve_D(inVec, buf, D, M, nBlocks_dot, nThreads_dot);
	
	MatrixType useDagger = MatrixType::Dagger;
	void *diagArgs[] = {(void*)&buf, (void*)&afterCG, (void*) &vol, (void*) &fermion_mass, (void*) &g_coupling, (void*)&useDagger, (void*)&M};
	// hopping should be passed to all the off-diagonal (in spacetime) functions: Deo, Doe
	void *hoppingArgs[] = {(void*)&buf, (void*) &afterCG, (void*) &lattice.vol, (void*) &useDagger, (void*) &lattice.IUP, (void*) &lattice.IDN};
	D.applyD(diagArgs, hoppingArgs);
	cudaDeviceSynchronize();

	// Set up dot product call
	void *driftArgs[] = {(void*) &inVec, (void*) &afterCG, (void*) &outVec, (void*) &vol};
	auto dimGrid = dim3(nBlocks_drift, 1, 1);
	auto dimBlock = dim3(nThreads_drift, 1, 1);

	cudaLaunchCooperativeKernel((void*)&computeDrift, dimGrid, dimBlock, driftArgs, 0, NULL);
	cudaDeviceSynchronize();
	
	cudaFree(afterCG);
	cudaFree(buf);
	 
}

__global__ void computeDrift(Spinor<double> *inVec, Spinor<double> *afterCG, cpdouble *outVec, int const vol){

	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();

	thrust::complex<double> im (0.0, 1.0);
	double const g_coupling = 0.1;

	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		// Drift for sigma
		outVec[i] = g_coupling * (	      conj(afterCG[i].val[0])*inVec[i].val[0]
										+ conj(afterCG[i].val[1])*inVec[i].val[1] 
										- conj(afterCG[i].val[2])*inVec[i].val[2] 
										+ conj(afterCG[i].val[3])*inVec[i].val[3]);

		// Drift for pi1
		outVec[i + vol] = im * g_coupling * (	- conj(afterCG[i].val[0])*inVec[i].val[3]
											 	+ conj(afterCG[i].val[1])*inVec[i].val[2] 
												+ conj(afterCG[i].val[2])*inVec[i].val[1] 
												+ conj(afterCG[i].val[3])*inVec[i].val[0]);

		// Drift for pi2
		outVec[i + 2*vol] = im * g_coupling * (	  im * conj(afterCG[i].val[0])*inVec[i].val[3] 
												- im * conj(afterCG[i].val[1])*inVec[i].val[2] 
												- im * conj(afterCG[i].val[2])*inVec[i].val[1] 
												+ im * conj(afterCG[i].val[3])*inVec[i].val[0]);

		// Drift for pi3
		outVec[i + 3*vol] = im * g_coupling * (	- conj(afterCG[i].val[0])*inVec[i].val[1]
												+ conj(afterCG[i].val[1])*inVec[i].val[0]
												+ conj(afterCG[i].val[2])*inVec[i].val[3]
												- conj(afterCG[i].val[3])*inVec[i].val[2]);

	}

}
