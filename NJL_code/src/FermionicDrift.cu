#include "FermionicDrift.cuh"


extern __constant__ double yukawa_coupling_gpu;
extern __constant__ thrust::complex<double> im_gpu;
extern __constant__ double cutFraction_gpu;

FermionicDrift::FermionicDrift(int const seed) : gen(rd()), dist(0.0, 1.0)
{
	cudaMallocManaged(&state, sizeof(curandState) * spinor_vol);

	int nBlocks = 0;
	int nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, setZero_kernel);
	cudaDeviceSynchronize();
  	dimGrid_zero = dim3(nBlocks, 1, 1); 
  	dimBlock_zero = dim3(nThreads, 1, 1); 

	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, fillNormalRND);
	cudaDeviceSynchronize();
  	dimGrid_rnd = dim3(nBlocks, 1, 1); 
  	dimBlock_rnd = dim3(nThreads, 1, 1); 

	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, random_setup_kernel);
	cudaDeviceSynchronize();
  	auto dimGrid_setup = dim3(nBlocks, 1, 1); 
  	auto dimBlock_setup = dim3(nThreads, 1, 1); 
	void* setupArgs[3] = {(void*) &seed, (void*) &state, (void*) &spinor_vol};
	cudaLaunchCooperativeKernel((void*)&random_setup_kernel, dimGrid_setup, dimBlock_setup, setupArgs, 0, NULL);
	cudaDeviceSynchronize();
	
  	
	setZeroArgs[0] = (void*) &afterCG.data();
    setZeroArgs[1] = (void*) &spinor_vol;

	driftArgs[0] = (void*) &afterCG.data();
    driftArgs[1] = (void*) &noiseVec.data();
    driftArgs[2] = (void*) &noiseVec.data();

	rndArgs[0] = (void*) &noiseVec.data();
	rndArgs[1] = (void*) &state;
	rndArgs[2] = (void*) &spinor_vol;
}

__global__ void random_setup_kernel(int const seed, curandState *state, int const vol) {
	cg::grid_group grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		curand_init(seed, i, 0, &state[i]);
	}
}

void FermionicDrift::getForce(double *outVec, DiracOP<double>& D, CGsolver& CG, dim3 dimGrid_drift, dim3 dimBlock_drift, int const Nvectors){
	
	for(int n = 0; n < Nvectors; n++) {
        cudaLaunchCooperativeKernel((void*)&fillNormalRND, dimGrid_rnd, dimBlock_rnd, rndArgs, 0, NULL);
        cudaDeviceSynchronize();
        
        // set some spinors to zero
        setZeroArgs[0] = (void*)&afterCG.data();
        cudaLaunchCooperativeKernel((void*)&setZero_kernel, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
        cudaDeviceSynchronize();

        CG.solve(noiseVec.data(), buf.data(), D, MatrixType::Dagger);
        D.applyD(buf.data(), afterCG.data(), MatrixType::Normal);
        cudaDeviceSynchronize();
                
        DriftState mode;
        driftArgs[0] = (void*) &afterCG.data();
        driftArgs[1] = (void*) &noiseVec.data();
        driftArgs[2] = (void*) &outVec;
        driftArgs[3] = (void*) &mode;
        driftArgs[4] = (void*) &Nvectors;
        
        if (n == 0) mode = DriftState::Init;
        else if (n == (Nvectors - 1)) mode = DriftState::End;
        else mode = DriftState::Other;
        
        cudaLaunchCooperativeKernel((void*)&computeDrift, dimGrid_drift, dimBlock_drift, driftArgs, 0, NULL);
        cudaDeviceSynchronize();
    }
	 
}


__global__ void computeDrift(cp<double> *afterCG,cp<double> *noise, double *outVec, DriftState const mode, int const Nvectors){

	cg::grid_group grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
        if (mode == DriftState::Init) outVec[i] = 0.0;
		outVec[i] += - yukawa_coupling_gpu * ( conj(afterCG[4*i+0])*noise[4*i+0]
                                            + conj(afterCG[4*i+1])*noise[4*i+1] 
                                            + conj(afterCG[4*i+2])*noise[4*i+2] 
                                            + conj(afterCG[4*i+3])*noise[4*i+3]).real();
        if (mode == DriftState::End) outVec[i] /= Nvectors;
	}

}

__global__ void fillNormalRND(cp<double>* vec, curandState *state, int const vol){
	cg::grid_group grid = cg::this_grid();
	auto myState = state[grid.thread_rank()];
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){	 
		myState = state[i]; 
		vec[i] = curand_normal_double(&myState); 
		state[i] = myState;
	}
}

