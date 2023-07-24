#include "FermionicDrift.cuh"


extern __constant__ double yukawa_coupling_gpu;
extern __constant__ thrust::complex<double> im_gpu;


FermionicDrift::FermionicDrift(int const seed) : gen(rd()), dist(0.0, 1.0)
{
    cudaMallocManaged(&afterCG, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&buf, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&noiseVec, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&eobuf, sizeof(thrust::complex<double>) * vol);
	cudaMallocManaged(&state, sizeof(curandState) * spinor_vol);

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
	
  	
	setZeroArgs[0] = (void*) &afterCG;
    setZeroArgs[1] = (void*) &spinor_vol;

	convArgs[0] = (void*) &eobuf;
    convArgs[1] = (void*) &eobuf;
    
	driftArgs[0] = (void*) &afterCG;
    driftArgs[1] = (void*) &noiseVec;
    driftArgs[2] = (void*) &noiseVec; 

	rndArgs[0] = (void*) &noiseVec;
	rndArgs[1] = (void*) &state;
	rndArgs[2] = (void*) &spinor_vol;

}

__global__ void random_setup_kernel(int const seed, curandState *state, int const vol) {
	cg::grid_group grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		curand_init(seed, i, 0, &state[i]);
	}
}

void FermionicDrift::getForce(double *outVec, DiracOP<double>& D, thrust::complex<double> *M, CGsolver& CG, dim3 dimGrid_drift, dim3 dimBlock_drift){
	
	cudaLaunchCooperativeKernel((void*)&fillNormalRND, dimGrid_rnd, dimBlock_rnd, rndArgs, 0, NULL);
	cudaDeviceSynchronize();
	
	// set some spinors to zero
	setZeroArgs[0] = (void*)&afterCG;
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();

	switch (CGmode){
		case '0':
			CG.solve(noiseVec, buf, D, MatrixType::Dagger);
			D.setInVec(buf);
			D.setOutVec(afterCG);
			D.setDagger(MatrixType::Normal);
			D.applyD();
			cudaDeviceSynchronize();
			break;
		case '1':
			
			//Dirac_d.D_oo_inv(psiField.pos.begin() + lattice.vol/2, temp2.begin());
			//Dirac_d.D_eo(temp2.begin(), temp1.begin());
			/*D.setInVec(); D.setOutVec();
			cudaLaunchCooperativeKernel((void*)&D_oo_inv<T>, dimGrid_Doo_inv, dimBlock_Doo_inv, diagArgs, 0, NULL);
    		cudaDeviceSynchronize();
			cudaLaunchCooperativeKernel((void*)&D_eo<T>, dimGrid_Deo, dimBlock_Deo, diagArgs, 0, NULL);
    		cudaDeviceSynchronize();

			spinorDiff(psiField.pos.begin(), psiField.pos.begin() + lattice.vol/2, temp1.begin(), temp2.begin());
			
			CGdouble.solve_Dhat(temp2.begin(), temp2.end(), temp3.begin());
			for(int i=0; i<lattice.vol/2; i++) std::fill(psiField.pos[i].val.begin(), psiField.pos[i].val.begin(), 0.0);
			Dirac_d.applyDhatTo(temp3.begin(), psiField.pos.begin(), MatrixType::Dagger);

			std::fill(temp1.begin(), temp1.end(), Spinor_d());
			Dirac_d.D_oe(psiField.pos.begin(), temp1.begin());

			spinorDiff(psiField.pos.begin() + lattice.vol/2, psiField.pos.end(), temp1.begin(), temp3.begin());

			for(int i=lattice.vol/2; i<lattice.vol; i++) std::fill(psiField.pos[i].val.begin(), psiField.pos[i].val.end(), 0.0);
			Dirac_d.D_oo_inv(temp3.begin(), psiField.pos.begin() + lattice.vol/2);*/
			break;
		}
		
	driftArgs[0] = (void*) &afterCG;
	driftArgs[1] = (void*) &noiseVec;
	driftArgs[2] = (void*) &outVec;
	cudaLaunchCooperativeKernel((void*)&computeDrift, dimGrid_drift, dimBlock_drift, driftArgs, 0, NULL);
	cudaDeviceSynchronize();
	
	
	/*convArgs[0] = (void*)&eobuf;
	convArgs[1] = (void*)&outVec;
	cudaLaunchCooperativeKernel((void*)&eoConv, dimGrid_conv, dimBlock_conv, convArgs, 0, NULL);
	cudaDeviceSynchronize();*/
	 
}

__global__ void eoConv(thrust::complex<double> *eoVec, double *normalVec){
	cg::grid_group grid = cg::this_grid();
	int eo_i;
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		eo_i = NormalToEO(i);
		normalVec[i] = eoVec[eo_i].real();
	}
}

__global__ void computeDrift(Spinor<double> *afterCG, Spinor<double> *noise, double *outVec){

	cg::grid_group grid = cg::this_grid();
	int eo_i;
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		eo_i = NormalToEO(i);
		outVec[i] = -yukawa_coupling_gpu * (  conj(afterCG[eo_i].val[0])*noise[eo_i].val[0]
											+ conj(afterCG[eo_i].val[1])*noise[eo_i].val[1] 
											- conj(afterCG[eo_i].val[2])*noise[eo_i].val[2] 
											+ conj(afterCG[eo_i].val[3])*noise[eo_i].val[3]).real();

	}

}

__global__ void fillNormalRND(thrust::complex<double>* vec, curandState *state, int const vol){
	cg::grid_group grid = cg::this_grid();
	auto myState = state[grid.thread_rank()];
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){	 
		myState = state[i]; 
		vec[i] = curand_normal_double(&myState); 
		state[i] = myState;
	}
}
