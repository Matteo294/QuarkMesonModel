#include <array>
#include <complex>
#include <iostream>
#include <thrust/complex.h>
#include <fstream>
#include <cooperative_groups.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "Dirac.cuh"
#include "Spinor.cuh"
#include "Lattice.cuh"
#include "params.h"

#include <chrono>

namespace cg = cooperative_groups;

//#define USE_EO_PRECONDITIONING

using std::conj;
using cpdouble = thrust::complex<double>;


cpdouble im {0.0, 1.0};

// Q: int4,float4 reading
// Q: extern ?
// Q: constants which are common to GPU and CPU?

// !! make CG a global function with dot, Dirac device functions. Moreover, buffers for CG could be allocated once for all.
// numBlocks, numThreads for Dirac could be determined once for all
// Note/Question: consider replacing cudaMallocManaged in the constructor with simple allocation and then use MemCpy to fill the buffers. Any significant speedup?
// remove force nthreads nblocs = 1 and always ask for 32 * size
// !!!!!!!!!!!!! g_coupling hardcoded!!!!! 
// make classes to avoid continuous allocation and deallocation

// VERY IMPORTANT NOTE: REMEMBER TO SET THE VECTOR TO ZERO BEFORE APPLYING ANY D_xx OR THE DIRAC OPERATOR !!!!

template <typename T>
__host__ void CGsolver_solve_D(Spinor<T> *inVec, Spinor<T> *outVec, DiracOP<T>& D, cpdouble *M, int const numBlocks, int const numThreads);

/*template <typename T>
__host__ void CGsolver_solve_Dhat(Spinor<T> *inVec, Spinor<T> *outVec, MesonsMat<T> *M, DiracOP<T> &D);*/

__global__ void gpuDotProduct(cpdouble *vecA, cpdouble *vecB, cpdouble *result, int size);

void mesonsToM(double *phi, cpdouble *M, int const vol){
	double *sigma, *pi1, *pi2, *pi3;
	sigma = phi;
	pi1 = phi + vol;
	pi2 = phi + 2*vol;
	pi3 = phi + 3*vol;
	for(int i=0; i<vol; i++){
		M[i] = sigma[i] + im * pi3[i];
		M[i + vol] = im * (pi1[i] - im*pi2[i]);
		M[i + 2*vol] = - im * (pi1[i] - im*pi2[i]);	
		M[i + 3*vol] = sigma[i] - im * pi3[i];
	}
}

// !!!!!!!!!!!!! THIS FUNCTION WORKS ONLY FOR REAL MESONS !!!!!!!!!!!!! 
void MtoMesons(cpdouble *M, double *phi, int const vol){
	double *sigma, *pi1, *pi2, *pi3;
	sigma = phi;
	pi1 = phi + vol;
	pi2 = phi + 2*vol;
	pi3 = phi + 3*vol;
	for(int i=0; i<vol; i++){
		sigma[i] = M[i].real();
		pi1[i] = M[i + vol].imag();
		pi2[i] = M[i + vol].real();
		pi3[i] = M[i].imag();
	}
}

		
int main() {

	cpdouble *M;
	Spinor<double> *in, *out;
	Lattice lattice(Nt, Nx);
	DiracOP<double> Dirac(fermion_mass, g_coupling, lattice);

	// Allocate two vectors and mesons matrix
	cudaMallocManaged(&M, sizeof(cpdouble) * 4 * lattice.vol);
	cudaMallocManaged(&in, sizeof(Spinor<double>) * lattice.vol);
	cudaMallocManaged(&out, sizeof(Spinor<double>) * lattice.vol);

	std::cout << Nt << " " << Nx << std::endl;
	
	// Set fields values
	for(int i=0; i<lattice.vol; i++){
		M[i] = sigma + im * pi[2];
		M[i + 3*lattice.vol] = sigma - im * pi[2];
		M[i + lattice.vol] = im * (pi[0] - im * pi[1]);
		M[i + 2*lattice.vol] = im * (pi[0] + im * pi[1]);
	}
	for(int i=0; i<lattice.vol; i++){in[i].setZero();}
	// set source
	in[0].val[0] = 1.0;
	in[0].val[1] = 1.0;
	in[0].val[2] = 1.0;
	in[0].val[3] = 1.0;

	MatrixType useDagger = MatrixType::Normal;
	// diagArgs should be passed to all the diagonal (in spacetime) functions: Doo, Dee, Dooinv, Deeinv
	void *diagArgs[] = {(void*)&in, (void*)&out, (void*) &lattice.vol, (void*) &fermion_mass, (void*) &g_coupling, (void*)&useDagger, (void*)&M};
	// hopping should be passed to all the off-diagonal (in spacetime) functions: Deo, Doe
	void *hoppingArgs[] = {(void*)&in, (void*) &out, (void*) &lattice.vol, (void*) &useDagger, (void*) &lattice.IUP, (void*) &lattice.IDN}; 
	
	int numBlocks = 0;
	int numThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, gpuDotProduct);
	cudaDeviceSynchronize();

	//numBlocks = 1;
	//numThreads = 1;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	CGsolver_solve_D(in, out, Dirac, M, numBlocks, numThreads);

	auto err = cudaGetLastError();
	std::cout << "Last error: " << err << " --> " << cudaGetErrorString(err) << "\n";

	for(int i=0; i<lattice.vol; i++){in[i].setZero();}
	useDagger = MatrixType::Dagger;
	diagArgs[0] = (void*) &out; diagArgs[1] = (void*) &in;
	hoppingArgs[0] = (void*) &out; hoppingArgs[1] = (void*) &in;
	Dirac.applyD(diagArgs, hoppingArgs);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "CG time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

	std::ofstream datafile;
	datafile.open("data.csv");
	datafile << "f0c0,f0c1,f1c0,f1c1" << "\n";
	cpdouble corr = 0.0;
	for(int nt=0; nt<Nt; nt++){
		corr = 0.0;
		for(int nx=0; nx<Nx; nx++){
			for(int j=0; j<4; j++) corr += in[lattice.toEOflat(nt, nx)].val[j];
		}
		datafile << corr.real() << "\n";
	}


	cudaFree(M);
	cudaFree(in);
	cudaFree(out);
	
	return 0;
}


template <typename T>
__host__ void CGsolver_solve_D(Spinor<T> *inVec, Spinor<T> *outVec, DiracOP<T>& D, cpdouble *M, int const numBlocks, int const numThreads){	
	
	int const vol = D.lattice.vol;
	int mySize = D.lattice.vol * 4;

	Spinor<T> *r, *p, *temp, *temp2; // allocate space ?? 
	thrust::complex<T> alpha; // allocate space ??
	T beta, rmodsq;
	cpdouble *dot_res;

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

	// Set up dot product call
	void *dotArgs[] = {(void*) &r, (void*) &r, (void*) &dot_res, (void*) &mySize};
	auto dimGrid = dim3(numBlocks, 1, 1);
	auto dimBlock = dim3(numThreads, 1, 1);

	cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * (32), NULL);
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
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * (32), NULL);
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
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * (32), NULL);
		cudaDeviceSynchronize();
		beta = dot_res->real() / rmodsq;

		// p = r - beta p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) p[i].val[j] = r[i].val[j] + beta*p[i].val[j];
		}

		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * (32), NULL);
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


/*
template <typename T>
__host__ void CGsolver_solve_Dhat(Spinor<T> *inVec, Spinor<T> *outVec, MesonsMat<T> *M, DiracOP<T> &D){	
	Spinor<T> *r, *p, *temp, *temp2; // allocate space
	thrust::complex<T> alpha, s; // allocate space
	T beta, rmodsq;
	
	int const vol = D.lattice.vol;

	cudaMallocManaged(&r, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&p, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&temp, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&temp2, sizeof(Spinor<T>) * vol);
	
	for(int i=0; i<vol; i++) {
		outVec[i] = Spinor<T> ();
		temp[i] = Spinor<T> ();
		temp2[i] = Spinor<T> ();
		for(int j=0; j<4; j++) r[i].val[j] = inVec[i].val[j];
		for(int j=0; j<4; j++) p[i].val[j] = r[i].val[j];
	}

	s = 0.0;
	for(int idx=0; idx<vol; idx++){
		s += r[idx].dot(r[idx]);
	}
	rmodsq = s.real();

	int k;
	for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) temp[i].val[j] = 2.0 * p[i].val[j];
		}

		D.applyDhat(p, temp2, M, MatrixType::Dagger);
		D.applyDhat(temp2, temp, M, MatrixType::Normal);
		
		s = 0.0;
		for(int idx=0; idx<vol; idx++){
			s += p[idx].dot(temp[idx]);
		}
		alpha = rmodsq / s; 

		// x += alpha p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) outVec[i].val[j] += alpha*p[i].val[j];
		}
		// r -= alpha A p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) r[i].val[j] -= alpha*temp[i].val[j];
		}

		
		s = 0.0;
		for(int idx=0; idx<vol; idx++){
			s += r[idx].dot(r[idx]);
		}
		beta = s.real() / rmodsq;

		// p = r - beta p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) p[i].val[j] = r[i].val[j] + beta*p[i].val[j];
		}

		s = 0.0;
		for(int idx=0; idx<vol; idx++){
			s += r[idx].dot(r[idx]);
		}
		rmodsq = s.real();
	}

	if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
	else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << sqrt(rmodsq) << "\n";

	cudaFree(r);
	cudaFree(p);
	cudaFree(temp);
	cudaFree(temp2);
}
*/












 /*#ifdef USE_EO_PRECONDITIONING

		Spinor<double> *temp1, *temp2, *temp3;
		cudaMallocManaged(&temp1, sizeof(Spinor<double>) * lattice.vol);
		cudaMallocManaged(&temp2, sizeof(Spinor<double>) * lattice.vol);
		cudaMallocManaged(&temp3, sizeof(Spinor<double>) * lattice.vol);

		for(int i=0; i<lattice.vol; i++){temp1[i].setZero(); temp2[i].setZero(); temp3[i].setZero();}

		Dirac.D_oo_inv_wrap(in, temp2, M, MatrixType::Normal);
		Dirac.D_eo_wrap(temp2, temp1, M, MatrixType::Normal);

		for(int i=0; i<lattice.vol/2; i++){for(int j=0; j<4; j++) temp2[i].val[j] = in[i].val[j] - temp1[i].val[j];}
		
		CGsolver_solve_Dhat(temp2, temp3, M, Dirac);

		for(int i=0; i<lattice.vol/2; i++) in[i].setZero();
		Dirac.applyDhat(temp3, in, M, MatrixType::Dagger);

		for(int i=0; i<lattice.vol; i++) temp1[i].setZero();
		Dirac.D_oe_wrap(in, temp1, M, MatrixType::Normal);

		for(int i=lattice.vol/2; i<lattice.vol; i++) {
			for(int j=0; j<4; j++) temp3[i].val[j] = in[i].val[j] - temp1[i].val[j];
		} 
		for(int i=lattice.vol/2; i<lattice.vol; i++) in[i].setZero();

		Dirac.D_oo_inv_wrap(temp3, in, M, MatrixType::Normal);

		cudaFree(temp1);
		cudaFree(temp2);
		cudaFree(temp3);		

	#else*/
