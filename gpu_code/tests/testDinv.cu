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


		
int main() {

	thrust::complex<double> *M;
	Spinor<double> *in, *out, *in_copy;
	Lattice lattice(Nt, Nx);
	DiracOP<double> Dirac(fermion_mass, g_coupling, lattice);

	// Allocate two vectors and mesons matrix
	cudaMallocManaged(&M, sizeof(thrust::complex<double>) * 4 * lattice.vol);
	cudaMallocManaged(&in, sizeof(Spinor<double>) * lattice.vol);
	cudaMallocManaged(&in_copy, sizeof(Spinor<double>) * lattice.vol);
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
//		in[i].val[0] = 1.0 * exp(im*idx[1]*q+im*idx[0]*p);
//		in[i].val[1] = 1.0 * exp(im*idx[1]*q+im*idx[0]*p);
		in[i].val[0] = 0.3*idx[1];// * exp(im*idx[1]*q+im*idx[0]*p);
		in[i].val[1] = 0.2*idx[0] + 0.1*idx[1];// * exp(im*idx[1]*q+im*idx[0]*p);
		for(int j=0; j<4; j++) in_copy[i].val[j] = in[i].val[j];
	}


	// Calculate Occupancy
	int nBlocks_dot = 0;
	int nThreads_dot = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks_dot, &nThreads_dot, gpuDotProduct);
	cudaDeviceSynchronize();
	std::cout << nBlocks_dot << '\t' << nThreads_dot << '\n';
	nBlocks_dot = 1;
	nThreads_dot = 1;

    // Apply Dinv
	CGsolver_solve_D(in, out, Dirac, M, nBlocks_dot, nThreads_dot);
	for(int i=0; i<lattice.vol; i++){in[i].setZero();}
	MatrixType useDagger = MatrixType::Dagger;
	// diagArgs should be passed to all the diagonal (in spacetime) functions: Doo, Dee, Dooinv, Deeinv
	void *diagArgs[] = {(void*)&out, (void*)&in, (void*) &lattice.vol, (void*) &fermion_mass, (void*) &g_coupling, (void*)&useDagger, (void*)&M};
	// hopping should be passed to all the off-diagonal (in spacetime) functions: Deo, Doe
	void *hoppingArgs[] = {(void*)&out, (void*) &in, (void*) &lattice.vol, (void*) &useDagger, (void*) &lattice.IUP, (void*) &lattice.IDN};
	Dirac.applyD(diagArgs, hoppingArgs);
    cudaDeviceSynchronize();

	std::ofstream myfile;
	myfile.open("planewave.csv");
	myfile << "nt,nx,v1,v2,v3,v4" << std::endl;
	int i;
	thrust::complex<double> v, w = 0.0;
	double pi_sq = pi[0]*pi[0] + pi[1]*pi[1] + pi[2]*pi[2];
	for(int nx=0; nx<Nx; nx++){
		for(int nt=0; nt<Nt; nt++){
			i = lattice.toEOflat(nt, nx);
			v = 		((fermion_mass + g_coupling*sigma + 2.0*sin(0.5*p)*sin(0.5*p) + 2.0*sin(0.5*q)*sin(0.5*q) - im*sin(p)) * in_copy[i].val[0] +
						(g_coupling*pi[2] - im*sin(q)) * in_copy[i].val[1]) /
						(pow(fermion_mass + g_coupling*sigma + 2.0*sin(0.5*p)*sin(0.5*p) + 2.0*sin(0.5*q)*sin(0.5*q), 2) + sin(q)*sin(q) + sin(p)*sin(p) + g_coupling*g_coupling*pi_sq);
			w = 		((fermion_mass + g_coupling*sigma + 2.0*sin(0.5*p)*sin(0.5*p) + 2.0*sin(0.5*q)*sin(0.5*q) + im*sin(p)) * in_copy[i].val[1] -
						(g_coupling*pi[2] + im*sin(q)) * in_copy[i].val[0]) /
						(pow(fermion_mass + g_coupling*sigma + 2.0*sin(0.5*p)*sin(0.5*p) + 2.0*sin(0.5*q)*sin(0.5*q), 2) + sin(q)*sin(q) + sin(p)*sin(p) + g_coupling*g_coupling*pi_sq);		
			myfile  	<< nt << "," << nx << "," << in[i].val[0].real() << "," << in[i].val[1].real() << ","
						<< v.real() << ","
						<< w.real() << "\n";
			std::cout 	<< "Site nt=" << nt << " nx=" << nx << "\n"  
						<< "1st component \t --> \t measured: " << in[i].val[0] << "\t\t expected :" << v << "\n"
						<< "2nd component \t --> \t measured: " << in[i].val[1] << "\t\t expected :" << w << "\n\n";

		}
	}

	for(int i=0; i<lattice.vol; i++){out[i].setZero();}
	useDagger = MatrixType::Normal;
	diagArgs[0] = (void*) &in; diagArgs[1] = (void*) &out;
	hoppingArgs[0] = (void*) &in; hoppingArgs[1] = (void*) &out;
	Dirac.applyD(diagArgs, hoppingArgs);
    cudaDeviceSynchronize();

	std::cout << "D Dinv psi = " << out[0].val[0] << "\t psi" << in_copy[0].val[0] << std::endl;
 
	std::cout << "Last error: " << cudaGetLastError() << ": " << cudaGetErrorString(cudaGetLastError()) << "\n";

	cudaFree(M);
	cudaFree(in);
	cudaFree(out);
	cudaFree(in_copy);
	
	return 0;
}


template <typename T>
__host__ void CGsolver_solve_D(Spinor<T> *inVec, Spinor<T> *outVec, DiracOP<T>& D, thrust::complex<double> *M, int const numBlocks, int const numThreads){	
	
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
	std::cout << "dot prod = " << *dot_res << '\n';
	rmodsq = dot_res->real();
	std::cout << *dot_res << " " << r[0].val[0] << "\n";

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
	std::cout << "pAp prod = " << *dot_res << '\n';
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
	std::cout << "dot prod = " << *dot_res << '\n';
		beta = abs(*dot_res) / rmodsq;

		// p = r - beta p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) p[i].val[j] = r[i].val[j] + beta*p[i].val[j];
		}

		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(cpdouble) * (32), NULL);
		cudaDeviceSynchronize();
	std::cout << "dot prod = " << *dot_res << "\n\n";
		rmodsq = abs(*dot_res);
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
