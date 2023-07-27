#include "CGsolver.cuh"


__device__ void applyD(cp<double> *in, cp<double> *out, int vol){
    auto grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		out[i] = ((cp<double>) i + 0.1) * in[i];
	}
}


CGsolver::CGsolver(){
	cudaMallocManaged(&dot_res, sizeof(thrust::complex<double>));
	cudaMallocManaged(&rmodsq, sizeof(double));
	cudaMallocManaged(&alpha, sizeof(thrust::complex<double>));
	cudaMallocManaged(&beta, sizeof(thrust::complex<double>));
        
	// Arguments are assigned in the following, but then the pointers are modified before each kernel call
	dotArgs[0] = (void*) &(r.data()); 
	dotArgs[1] = (void*) &(r.data()); 
	dotArgs[2] = (void*) &dot_res; 
	dotArgs[3] = (void*) &myvol;
	
	setZeroArgs[0] = (void*) &temp.data(); 
	setZeroArgs[1] = (void*) &myvol;
	
	sumArgs[0] = (void*) &r.data(); 
	sumArgs[1] = (void*) &r.data(); 
	sumArgs[2] = (void*) &r.data(); 
	sumArgs[3] = (void*) &beta;
	sumArgs[4] = (void*) &myvol;
    
	copyArgs[0] = (void*) &r.data(); 
	copyArgs[1] = (void*) &r.data(); 
	copyArgs[2] = (void*) &myvol;

	myvol = 4*vol;

	
}

void CGsolver::solve(cp<double>  *inVec, cp<double> *outVec, DiracOP<double>& D, MatrixType Mtype){
    auto MatType = Mtype;
    int nBlocks = 0;
	int nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, solve_kernel);
	cudaDeviceSynchronize();
	auto dimGrid = dim3(nBlocks, 1, 1);
	auto dimBlock = dim3(nThreads, 1, 1);
    int sMemSize = sizeof(thrust::complex<double>) * (nThreads/32);
    void *solveArgs[] = {(void*) &inVec, (void*) &outVec, 
						(void*) &temp.data(), (void*) &temp2.data(), (void*) &r.data(), (void*) &p.data(), 
						(void*) &alpha, (void*)&beta,
						(void*) &D.M, 
						(void*) &D.EO2N, (void*) &D.IUP, (void*)&D.IDN,
						(void*) &MatType, (void*)&dot_res, (void*)&rmodsq};
    cudaLaunchCooperativeKernel((void*) solve_kernel, dimGrid, dimBlock, solveArgs, sMemSize, NULL);
	cudaDeviceSynchronize();
}

__device__ void setZeroGPU(thrust::complex<double> *v, int const vol){
	cg::grid_group grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < vol; i += grid.size()) v[i] = 0.0;
}

__device__ void copyVec(thrust::complex<double> *v1,thrust::complex<double> *v2, int const vol){
	cg::grid_group grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < vol; i += grid.size()) v1[i] = v2[i];
}


__global__ void solve_kernel(cp<double>  *inVec, cp<double> *outVec, 
                             cp<double> *temp, cp<double> *temp2, cp<double> *r, cp<double> *p,
							 cp<double> *alpha, cp<double> *beta,
							 double *M, 
							 int *EO2N, my2dArray *IUP, my2dArray *IDN,  
							 MatrixType Mtype, cp<double> *dot_res, double *rmodsq)
{
        
    auto grid = cg::this_grid();
	int myvol = 4*vol;
    
    setZeroGPU(outVec, myvol);
    cg::sync(grid);
    setZeroGPU(temp, myvol);
    cg::sync(grid);
    setZeroGPU(temp2, myvol);
    cg::sync(grid);
    
    copyVec(r, inVec, myvol);
    cg::sync(grid);
    copyVec(p, inVec, myvol);
    cg::sync(grid);
    

	if (threadIdx.x == 0 && blockIdx.x == 0) *dot_res = 0.0;
    gpuDotProduct(r, r, dot_res, myvol);
    cg::sync(grid);

    *rmodsq = abs(*dot_res);

	auto MatType = Mtype;

	cg::sync(grid); 

    int k;
	for(k=0; k<IterMax && sqrt(*rmodsq) > tolerance; k++){

		setZeroGPU(temp, myvol);
        cg::sync(grid);
        setZeroGPU(temp2, myvol);

  		// Apply D dagger
		if (Mtype == MatrixType::Normal) MatType = MatrixType::Dagger;
		else MatType = MatrixType::Normal;
        cg::sync(grid);
        D_oo(p, temp2, MatType, M, EO2N);
        cg::sync(grid);
        D_ee(p, temp2, MatType, M, EO2N);
        cg::sync(grid);
        D_eo(p, temp2, MatType, IUP, IDN);
        cg::sync(grid);
        D_oe(p, temp2, MatType, IUP, IDN);

		// Apply D
		if (Mtype == MatrixType::Normal) MatType = MatrixType::Normal;
		else MatType = MatrixType::Dagger;
        cg::sync(grid);
        D_oo(temp2, temp, MatType, M, EO2N);
        cg::sync(grid);
        D_ee(temp2, temp, MatType, M, EO2N);
        cg::sync(grid);
        D_eo(temp2, temp, MatType, IUP, IDN);
        cg::sync(grid);
        D_oe(temp2, temp, MatType, IUP, IDN);

    
		if (threadIdx.x == 0 && blockIdx.x == 0) *dot_res = 0.0;
        cg::sync(grid);
        gpuDotProduct(p, temp, dot_res, myvol);
        cg::sync(grid);
        
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			*alpha = *rmodsq / *dot_res; 
		}


		// x = x + alpha p
		cg::sync(grid);
        gpuSumSpinors(outVec, p, outVec, *alpha, myvol);
		
		// r = r - alpha A p
		if (threadIdx.x == 0 && blockIdx.x == 0) *alpha = - *alpha;
		cg::sync(grid);
        gpuSumSpinors(r, temp, r, *alpha, myvol);

        if (threadIdx.x == 0 && blockIdx.x == 0) *dot_res = 0.0;
		cg::sync(grid);
        gpuDotProduct(r, r, dot_res, myvol);
        cg::sync(grid);
        
		if (threadIdx.x == 0 && blockIdx.x == 0){
			*beta = abs(*dot_res) / *rmodsq;
        	*rmodsq = abs(*dot_res);
		}

		// p = r - beta p
		cg::sync(grid);
        gpuSumSpinors(r, p, p, *beta, myvol);
        cg::sync(grid);

	
	}

	//if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
	//else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << sqrt(rmodsq) << "\n";
	//if (k >= IterMax) std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << sqrt(rmodsq) << "\n";
}

__device__ void gpuSumSpinors(cp<double> *s1, cp<double> *s2, cp<double> *res, thrust::complex<double> c, int size){
	auto grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < size; i += grid.size()){
		res[i] = s1[i] + c * s2[i];
	}
}





/*void CGsolver::solveEO(cp<double>  *inVec, cp<double> *outVec, DiracOP<double>& D, MatrixType Mtype){
	double rmodsq;

	myvol = 2*vol;
	setZeroArgs[0] = (void*) &(temp.data());
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();
	setZeroArgs[0] = (void*) &temp2.data();
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();
	setZeroArgs[0] = (void*) &outVec;
	cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
	cudaDeviceSynchronize();
    copyArgs[0] = (void*) &r.data(); copyArgs[1] = (void*) &inVec;
	cudaLaunchCooperativeKernel((void*)&copyVec, dimGrid_zero, dimBlock_copy, copyArgs, 0, NULL);
	cudaDeviceSynchronize();
    copyArgs[0] = (void*) &p.data(); copyArgs[1] = (void*) &r.data();
	cudaLaunchCooperativeKernel((void*)&copyVec, dimGrid_zero, dimBlock_copy, copyArgs, 0, NULL);
	cudaDeviceSynchronize();

	*dot_res = 0.0;
	cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
	cudaDeviceSynchronize();
	rmodsq = abs(*dot_res);


	int k;
	for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

		setZeroArgs[0] = (void*) &temp.data();
		cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
		cudaDeviceSynchronize();
		setZeroArgs[0] = (void*) &temp2.data();
		cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
		cudaDeviceSynchronize();

		// Apply D dagger
		if (Mtype == MatrixType::Normal) D.setDagger(MatrixType::Dagger);
		else D.setDagger(MatrixType::Normal);
		D.setInVec(p.data());
		D.setOutVec(temp2.data());
		D.applyDhat();
		// Apply D
		if (Mtype == MatrixType::Normal) D.setDagger(MatrixType::Normal);
		else D.setDagger(MatrixType::Dagger);
		D.setInVec(temp2.data());
		D.setOutVec(temp.data());
		D.applyDhat();
		
		dotArgs[0] = (void*) &p.data(); dotArgs[1] = (void*) &temp.data();

		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
		cudaDeviceSynchronize();
		alpha = rmodsq / *dot_res; 


		// x = x + alpha p
		sumArgs[0] = (void*) &outVec;
		sumArgs[1] = (void*) &p.data();
		sumArgs[2] = (void*) &outVec;
		sumArgs[3] = (void*) &alpha;
		cudaLaunchCooperativeKernel((void*)&gpuSumSpinors, dimGrid_dot, dimBlock_dot, sumArgs, 0, NULL);
		cudaDeviceSynchronize();
		
		// r = r - alpha A p
		sumArgs[0] = (void*) &r.data();
		sumArgs[1] = (void*) &temp.data();
		sumArgs[2] = (void*) &r.data();
		sumArgs[3] = (void*) &alpha;
		alpha = -alpha;
		cudaLaunchCooperativeKernel((void*)&gpuSumSpinors, dimGrid_dot, dimBlock_dot, sumArgs, 0, NULL);
		cudaDeviceSynchronize();


		dotArgs[0] = (void*) &r.data(); dotArgs[1] = (void*) &r.data();
		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid_dot, dimBlock_dot, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
		cudaDeviceSynchronize();
		beta = abs(*dot_res) / rmodsq;
        rmodsq = abs(*dot_res);

		// p = r - beta p
		sumArgs[0] = (void*) &r.data();
		sumArgs[1] = (void*) &p.data();
		sumArgs[2] = (void*) &p.data();
		sumArgs[3] = (void*) &beta;
		cudaLaunchCooperativeKernel((void*)&gpuSumSpinors, dimGrid_dot, dimBlock_dot, sumArgs, 0, NULL);
		cudaDeviceSynchronize();
	
	}

	//if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
	//else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << sqrt(rmodsq) << "\n";
	if (k >= IterMax) std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << sqrt(rmodsq) << "\n";
}*/

/*__global__ void gpuDotProduct(thrust::complex<double> *vecA, thrust::complex<double> *vecB, thrust::complex<double> *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
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


__global__ void gpuSumSpinors(cp<double> *s1, cp<double> *s2, cp<double> *res, thrust::complex<double> c, int size){
	auto grid = cg::this_grid();
	for (int i = grid.thread_rank(); i < size; i += grid.size()){
		res[i] = s1[i] + c * s2[i];
	}
}*/
