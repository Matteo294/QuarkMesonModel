#include "CGsolver.cuh"


CGsolver::CGsolver(){
    cudaMallocManaged(&r, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&p, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&temp, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&temp2, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&dot_res, sizeof(thrust::complex<double>));

    int nBlocks = 0;
	int nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, gpuDotProduct);
	cudaDeviceSynchronize();
	dimGrid_dot = dim3(nBlocks, 1, 1);
	dimBlock_dot = dim3(nThreads, 1, 1);
    
    nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, setZeroGPU);
	cudaDeviceSynchronize();
    dimGrid_zero = dim3(nBlocks, 1, 1);
	dimBlock_zero = dim3(nThreads, 1, 1);
    
	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, gpuSumSpinors);
	cudaDeviceSynchronize();
	dimGrid_sum = dim3(nBlocks, 1, 1);
	dimBlock_sum = dim3(nThreads, 1, 1);
    
    nBlocks = 0;
	nThreads = 0;
	dimGrid_copy = dim3(nBlocks, 1, 1);
	dimBlock_copy = dim3(nThreads, 1, 1);
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, copyVec);
	cudaDeviceSynchronize();
	dimGrid_copy = dim3(nBlocks, 1, 1);
	dimBlock_copy = dim3(nThreads, 1, 1);

	
}

CGsolver::~CGsolver(){
	cudaFree(r);
	cudaFree(p);
	cudaFree(temp);
	cudaFree(temp2);
	cudaFree(dot_res);
}



void CGsolver::solve(Spinor<double>  *inVec, Spinor<double> *outVec, DiracOP<double>& D, MatrixType Mtype){
	double rmodsq;
    
    void *dotArgs[] = {(void*) &r, (void*) &r, (void*) &dot_res, (void*) &spinor_vol};
	void *setZeroArgs[] = {(void*)temp, (void*) &spinor_vol};
	void *sumArgs[] = {(void*) &r, (void*) &r, (void*) &r, (void*) &beta};
    void *copyArgs[] = {(void*) &r, (void*) &inVec, (void*) &spinor_vol};

	setZeroArgs[0] = (void*) &temp;
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();
	setZeroArgs[0] = (void*) &temp2;
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();
	setZeroArgs[0] = (void*) &outVec;
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();
    copyArgs[0] = (void*) &r; copyArgs[1] = (void*) &inVec;
	cudaLaunchCooperativeKernel((void*)&copyVec, dimGrid_zero, dimBlock_copy, copyArgs, 0, NULL);
	cudaDeviceSynchronize();
    copyArgs[0] = (void*) &p; copyArgs[1] = (void*) &r;
	cudaLaunchCooperativeKernel((void*)&copyVec, dimGrid_zero, dimBlock_copy, copyArgs, 0, NULL);
	cudaDeviceSynchronize();


	*dot_res = 0.0;
	cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
	cudaDeviceSynchronize();
	rmodsq = abs(*dot_res);


	int k;
	for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

		setZeroArgs[0] = (void*) &temp;
		cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
		cudaDeviceSynchronize();
		setZeroArgs[0] = (void*) &temp2;
		cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
		cudaDeviceSynchronize();

		// Apply D dagger
		if (Mtype == MatrixType::Normal) D.setDagger(MatrixType::Dagger);
		else D.setDagger(MatrixType::Normal);
		D.setInVec(p);
		D.setOutVec(temp2);
		D.applyD();
		// Apply D
		if (Mtype == MatrixType::Normal) D.setDagger(MatrixType::Normal);
		else D.setDagger(MatrixType::Dagger);
		D.setInVec(temp2);
		D.setOutVec(temp);
		D.applyD();
		
		dotArgs[0] = (void*) &p; dotArgs[1] = (void*) &temp;

		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
		cudaDeviceSynchronize();
		alpha = rmodsq / *dot_res; 


		// x = x + alpha p
		sumArgs[0] = (void*) &outVec;
		sumArgs[1] = (void*) &p;
		sumArgs[2] = (void*) &outVec;
		sumArgs[3] = (void*) &alpha;
		cudaLaunchCooperativeKernel((void*)&gpuSumSpinors, dimGrid_dot, dimBlock_dot, sumArgs, 0, NULL);
		cudaDeviceSynchronize();
		
		// r = r - alpha A p
		sumArgs[0] = (void*) &r;
		sumArgs[1] = (void*) &temp;
		sumArgs[2] = (void*) &r;
		sumArgs[3] = (void*) &alpha;
		alpha = -alpha;
		cudaLaunchCooperativeKernel((void*)&gpuSumSpinors, dimGrid_dot, dimBlock_dot, sumArgs, 0, NULL);
		cudaDeviceSynchronize();


		dotArgs[0] = (void*) &r; dotArgs[1] = (void*) &r;
		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
		cudaDeviceSynchronize();
		beta = abs(*dot_res) / rmodsq;
        rmodsq = abs(*dot_res);

		// p = r - beta p
		sumArgs[0] = (void*) &r;
		sumArgs[1] = (void*) &p;
		sumArgs[2] = (void*) &p;
		sumArgs[3] = (void*) &beta;
		cudaLaunchCooperativeKernel((void*)&gpuSumSpinors, dimGrid_dot, dimBlock_dot, sumArgs, 0, NULL);
		cudaDeviceSynchronize();
	
	}

	//if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
	//else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << sqrt(rmodsq) << "\n";

}

__global__ void gpuDotProduct(thrust::complex<double> *vecA, thrust::complex<double> *vecB, thrust::complex<double> *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	//*result = 0.0;
	extern __shared__ thrust::complex<double> tmp2[];

	thrust::complex<double> temp_sum = 0.0;
	for (int i = grid.thread_rank(); i < size; i += grid.size()) {
		temp_sum += conj(vecA[i]) * vecB[i];
	}

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	temp_sum = cg::reduce(tile32, temp_sum, cg::plus<thrust::complex<double>>());

	if (tile32.thread_rank() == 0) {
		tmp2[tile32.meta_group_rank()] = temp_sum;
	}

	cg::sync(cta);

	if (tile32.meta_group_rank() == 0) {
		temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp2[tile32.thread_rank()] : 0.0;
		temp_sum = cg::reduce(tile32, temp_sum, cg::plus<thrust::complex<double>>());

		if (tile32.thread_rank() == 0) {
		atomicAdd(reinterpret_cast<double*>(result), temp_sum.real());
		atomicAdd(reinterpret_cast<double*>(result)+1, temp_sum.imag());
		}
	}
}


__global__ void gpuSumSpinors(Spinor<double> *s1, Spinor<double> *s2, Spinor<double> *res, thrust::complex<double> c){
	auto grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		for(int j=0; j<4; j++) res[i].val[j] = s1[i].val[j] + c * s2[i].val[j];
	}
}
