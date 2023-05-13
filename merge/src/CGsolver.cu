#include "CGsolver.cuh"

CGsolver::CGsolver(){
    cudaMallocManaged(&r, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&p, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&temp, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&temp2, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&sol, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&dot_res, sizeof(thrust::complex<double>));

    int numBlocks = 0;
	int numThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, gpuDotProduct);
	cudaDeviceSynchronize();
	dimGrid_dot = dim3(numBlocks, 1, 1);
	dimBlock_dot = dim3(numThreads, 1, 1);

	/*numBlocks = 0;
	numThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, setSpinorToZero);
	cudaDeviceSynchronize();
	dimGrid_setZero = dim3(numBlocks, 1, 1);
	dimBlock_setZero = dim3(numThreads, 1, 1);*/

	
}

CGsolver::~CGsolver(){
	cudaFree(r);
	cudaFree(p);
	cudaFree(temp);
	cudaFree(temp2);
	cudaFree(dot_res);
	cudaFree(sol);
}




void CGsolver::solve(Spinor<double>  *inVec, Spinor<double> *outVec, DiracOP<double>& D, thrust::complex<double> *M){
	//cudaMemcpy(inVec, r, vol * sizeof(Spinor<double>), cudaMemcpyHostToDevice);
	thrust::complex<double> alpha; // allocate space ??
	double beta, rmodsq;


	int const mySize = 4 * vol;

	void *dotArgs[] = {(void*) &r, (void*) &r, (void*) &dot_res, (void*) &mySize};
	// set up set spinor to zero
	int nBlocks_zero = 0;
	int nThreads_zero = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks_zero, &nThreads_zero, setZeroGPU);
	cudaDeviceSynchronize();
	int const spinor_vol = 4*vol;
	void *setZeroArgs[] = {(void*)temp, (void*) &spinor_vol};

	D.setInVec(p);
	D.setOutVec(temp2);

	for(int i=0; i<vol; i++) {
		outVec[i] = Spinor<double> ();
		temp[i] = Spinor<double> ();
		temp2[i] = Spinor<double> ();
		for(int j=0; j<4; j++){
			r[i].val[j] = inVec[i].val[j];
			p[i].val[j] = r[i].val[j];
		}
	}

	setZeroArgs[0] = (void*) &temp;
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dim3(nBlocks_zero, 1, 1), dim3(nThreads_zero, 1, 1), setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();
	setZeroArgs[0] = (void*) &temp2;
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dim3(nBlocks_zero, 1, 1), dim3(nThreads_zero, 1, 1), setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();
	setZeroArgs[0] = (void*) &sol;
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dim3(nBlocks_zero, 1, 1), dim3(nThreads_zero, 1, 1), setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();


	*dot_res = 0.0;
	cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
	cudaDeviceSynchronize();
	rmodsq = dot_res->real();


	int k;
	for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

		// Set buffers to zero to store the result fo the Dirac operator applied to p
		setZeroArgs[0] = (void*) &temp;
		cudaLaunchCooperativeKernel((void*)&setZeroGPU, dim3(nBlocks_zero, 1, 1), dim3(nThreads_zero, 1, 1), setZeroArgs, 0, NULL);
		cudaDeviceSynchronize();
		setZeroArgs[0] = (void*) &temp2;
		cudaLaunchCooperativeKernel((void*)&setZeroGPU, dim3(nBlocks_zero, 1, 1), dim3(nThreads_zero, 1, 1), setZeroArgs, 0, NULL);
		cudaDeviceSynchronize();

		// Apply D dagger
		D.setDagger(MatrixType::Dagger);
		D.setInVec(p);
		D.setOutVec(temp2);
		D.applyD();
		// Apply D
		D.setDagger(MatrixType::Normal);
		D.setInVec(temp2);
		D.setOutVec(temp);
		D.applyD();
		
		dotArgs[0] = (void*) &p; dotArgs[1] = (void*) &temp;

		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
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
		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
		cudaDeviceSynchronize();
		beta = abs(*dot_res) / rmodsq;

		// p = r - beta p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) p[i].val[j] = r[i].val[j] + beta*p[i].val[j];
		}

		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
		cudaDeviceSynchronize();
		rmodsq = abs(*dot_res);
	}

	//if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
	//else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << sqrt(rmodsq) << "\n";

}

__global__ void gpuDotProduct(thrust::complex<double> *vecA, thrust::complex<double> *vecB, thrust::complex<double> *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	//*result = 0.0;
	extern __shared__ thrust::complex<double> tmp[];

	thrust::complex<double> temp_sum = 0.0;
	for (int i = grid.thread_rank(); i < size; i += grid.size()) {
		temp_sum += conj(vecA[i]) * vecB[i];
	}

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	temp_sum = cg::reduce(tile32, temp_sum, cg::plus<thrust::complex<double>>());

	if (tile32.thread_rank() == 0) {
		tmp[tile32.meta_group_rank()] = temp_sum;
	}

	cg::sync(cta);

	if (tile32.meta_group_rank() == 0) {
		temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
		temp_sum = cg::reduce(tile32, temp_sum, cg::plus<thrust::complex<double>>());

		if (tile32.thread_rank() == 0) {
		atomicAdd(reinterpret_cast<double*>(result), temp_sum.real());
		atomicAdd(reinterpret_cast<double*>(result)+1, temp_sum.imag());
		}
	}
}
