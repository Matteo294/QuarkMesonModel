#include "FermionicDrift.cuh"

extern __constant__ double yukawa_coupling_gpu;


FermionicDrift::FermionicDrift() : gen(rd()), dist(0.0, 1.0)
{
    cudaMallocManaged(&afterCG, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&buf, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&noiseVec, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&eobuf, sizeof(thrust::complex<double>) * 4 * vol);

	int nBlocks = 0;
	int nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, setZeroGPU);
	cudaDeviceSynchronize();
  	dimGrid_zero = dim3(nBlocks, 1, 1); 
  	dimBlock_zero = dim3(nThreads, 1, 1); 

	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, eoConv);
	cudaDeviceSynchronize();
  	dimGrid_conv = dim3(nBlocks, 1, 1); 
  	dimBlock_conv = dim3(nThreads, 1, 1); 
  	
	setZeroArgs[0] = (void*) &afterCG;
    setZeroArgs[1] = (void*) &spinor_vol;

	convArgs[0] = (void*) &eobuf;
    convArgs[1] = (void*) &eobuf;
    
	driftArgs[0] = (void*) &afterCG;
    driftArgs[1] = (void*) &noiseVec;
    driftArgs[2] = (void*) &noiseVec; 
}

void FermionicDrift::getForce(double *outVec, DiracOP<double>& D, thrust::complex<double> *M, CGsolver& CG, dim3 dimGrid_drift, dim3 dimBlock_drift){
	
	for(int i=0; i<vol; i++){ 
		for(int j=0; j<4; j++) noiseVec[i].val[j] = dist(gen);
	}
	
	// set some spinors to zero
	setZeroArgs[0] = (void*)&afterCG;
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();

	CG.solve(noiseVec, buf, D, MatrixType::Dagger);
	
	D.setInVec(buf);
	D.setOutVec(afterCG);
	D.setDagger(MatrixType::Normal);
	D.applyD();
	cudaDeviceSynchronize();

	driftArgs[0] = (void*) &afterCG;
    driftArgs[1] = (void*) &noiseVec;
    driftArgs[2] = (void*) &eobuf;
	cudaLaunchCooperativeKernel((void*)&computeDrift, dimGrid_drift, dimBlock_drift, driftArgs, 0, NULL);
	cudaDeviceSynchronize();
	
	convArgs[0] = (void*)&eobuf;
	convArgs[1] = (void*)&outVec;
	cudaLaunchCooperativeKernel((void*)&eoConv, dimGrid_conv, dimBlock_conv, convArgs, 0, NULL);
	cudaDeviceSynchronize();
	 
}

__global__ void eoConv(thrust::complex<double> *eoVec, double *normalVec){
	cg::grid_group grid = cg::this_grid();
	int eo_i;
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		eo_i = NormalToEO(i);
		normalVec[i] = eoVec[eo_i].real();
		normalVec[i + vol] = eoVec[eo_i + vol].real();
		normalVec[i + 2*vol] = eoVec[eo_i + 2*vol].real();
		normalVec[i + 3*vol] = eoVec[eo_i + 3*vol].real();
	}
}

__global__ void computeDrift(Spinor<double> *afterCG, Spinor<double> *noise, thrust::complex<double> *outVec){

	cg::grid_group grid = cg::this_grid();

	thrust::complex<double> im (0.0, 1.0);

	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		// Drift for sigma !!!!!!!!!!!!!!!! remember that this assumes outVec not even ordered !!!!!!!!!!!!
		outVec[i] = -yukawa_coupling_gpu * (	  conj(afterCG[i].val[0])*noise[i].val[0]
											+ conj(afterCG[i].val[1])*noise[i].val[1] 
											- conj(afterCG[i].val[2])*noise[i].val[2] 
											+ conj(afterCG[i].val[3])*noise[i].val[3]);

		// Drift for pi1
		outVec[i + vol] = -yukawa_coupling_gpu * (	- conj(afterCG[i].val[0])*noise[i].val[3]
											 		+ conj(afterCG[i].val[1])*noise[i].val[2] 
													- conj(afterCG[i].val[2])*noise[i].val[1] 
													+ conj(afterCG[i].val[3])*noise[i].val[0]);

		// Drift for pi2
		outVec[i + 2*vol] = -yukawa_coupling_gpu * (	  im * conj(afterCG[i].val[0])*noise[i].val[3] 
													- im * conj(afterCG[i].val[1])*noise[i].val[2] 
													- im * conj(afterCG[i].val[2])*noise[i].val[1] 
													+ im * conj(afterCG[i].val[3])*noise[i].val[0]);

		// Drift for pi3
		outVec[i + 3*vol] = -yukawa_coupling_gpu * (	- conj(afterCG[i].val[0])*noise[i].val[1]
													+ conj(afterCG[i].val[1])*noise[i].val[0]
													+ conj(afterCG[i].val[2])*noise[i].val[3]
													- conj(afterCG[i].val[3])*noise[i].val[2]);

	}

}
