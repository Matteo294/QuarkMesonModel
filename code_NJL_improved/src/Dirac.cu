#include "Dirac.cuh"

extern __constant__ double yukawa_coupling_gpu;
extern __constant__ double fermion_mass_gpu;
extern __constant__ thrust::complex<double> im_gpu;

template <typename T>
__host__ DiracOP<T>::DiracOP() : inVec(nullptr), outVec(nullptr), M(nullptr)
	{
        
		int numBlocks = 0;
        int numThreads = 0;
        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, D_ee<T>);
        dimGrid_Dee = dim3(numBlocks, 1, 1);    
        dimBlock_Dee = dim3(numThreads, 1, 1);      

        numBlocks = 0;
        numThreads = 0;
        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, D_oo<T>);
        dimGrid_Doo = dim3(numBlocks, 1, 1);    
        dimBlock_Doo = dim3(numThreads, 1, 1);
        
        numBlocks = 0;
        numThreads = 0;
        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, D_eo<T>);
        dimGrid_Deo = dim3(numBlocks, 1, 1);    
        dimBlock_Deo = dim3(numThreads, 1, 1);
        
        numBlocks = 0;
        numThreads = 0;
        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, D_oe<T>);
        dimGrid_Doe = dim3(numBlocks, 1, 1);    
        dimBlock_Doe = dim3(numThreads, 1, 1);


		auto idx = eoToVec(0);
		for(int i=0; i<vol; i++){
			idx = eoToVec(i);
			IUP.at[i][0] = toEOflat(PBC(idx[0]+1, Sizes[0]), idx[1]);
			IUP.at[i][1] = toEOflat(idx[0], PBC(idx[1]+1, Sizes[1]));
			IDN.at[i][0] = toEOflat(PBC(idx[0]-1, Sizes[0]), idx[1]);
			IDN.at[i][1] = toEOflat(idx[0], PBC(idx[1]-1, Sizes[1]));
		}

		diagArgs[0] = (void*) &inVec;
		diagArgs[1] = (void*) &outVec;
		diagArgs[2] = (void*) &useDagger;
		diagArgs[3] = (void*) &M;
		hoppingArgs[0] = (void*) &inVec;
		hoppingArgs[1] = (void*) &outVec;
		hoppingArgs[2] = (void*) &useDagger;
		hoppingArgs[3] = (void*) &IUP.at;
		hoppingArgs[4] = (void*) &IDN.at;      
    }


template <typename T>
__host__ void DiracOP<T>::applyD(){

    cudaLaunchCooperativeKernel((void*)&D_ee<T>, dimGrid_Dee, dimBlock_Dee, diagArgs, 0, NULL);
    cudaDeviceSynchronize();
    
    cudaLaunchCooperativeKernel((void*)&D_oo<T>, dimGrid_Doo, dimBlock_Doo, diagArgs, 0, NULL);
    cudaDeviceSynchronize();

 	cudaLaunchCooperativeKernel((void*)&D_eo<T>, dimGrid_Deo, dimBlock_Deo, hoppingArgs, 0, NULL);
    cudaDeviceSynchronize();

    cudaLaunchCooperativeKernel((void*)&D_oe<T>, dimGrid_Doe, dimBlock_Doe, hoppingArgs, 0, NULL);
    cudaDeviceSynchronize();

}


template <typename T>
__global__ void D_oo(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, cp<T> *M){

    auto grid = cg::this_grid();

    thrust::complex<T> const g = static_cast<thrust::complex<T>> (yukawa_coupling_gpu);
    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> mass = static_cast<thrust::complex<T>> (fermion_mass_gpu);
    thrust::complex<T> half {0.5, 0.0};

    for (int i = grid.thread_rank() + vol/2; i < vol; i += grid.size()) {
        if (useDagger == MatrixType::Dagger){
            outVec[4*i]     += (two + mass + g * conj(*M)) * inVec[i];
            outVec[4*i+1]   += (two + mass + g * conj(*M)) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * conj(*M)) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * conj(*M)) * inVec[4*i+3];
        } else{
            outVec[4*i]     += (two + mass + g * (*M)) * inVec[4*i];
            outVec[4*i+1]   += (two + mass + g * (*M)) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * (*M)) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * (*M)) * inVec[4*i]+3;
        }
    }
}

template <typename T>
__global__ void D_ee(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, cp<T> *M){

    auto grid = cg::this_grid();

    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> const g = static_cast<thrust::complex<T>> (yukawa_coupling_gpu);
    thrust::complex<T> const mass = static_cast<thrust::complex<T>> (fermion_mass_gpu);
    thrust::complex<T> half {0.5, 0.0};

    for (int i = grid.thread_rank(); i < vol/2; i += grid.size()) {
    
    if (useDagger == MatrixType::Dagger){
        outVec[4*i]     += (two + mass + g * conj(*M)) * inVec[4*i];
        outVec[4*i+1]   += (two + mass + g * conj(*M)) * inVec[4*i+1];
        outVec[4*i+2]   += (two + mass + g * conj(*M)) * inVec[4*i+2];
        outVec[4*i+3]   += (two + mass + g * conj(*M)) * inVec[4*i+3];
    } else {
        outVec[4*i]     += (two + mass + g * (*M)) * inVec[4*i];
        outVec[4*i+1]   += (two + mass + g * (*M)) * inVec[4*i+1];
        outVec[4*i+2]   += (two + mass + g * (*M)) * inVec[4*i+2];
        outVec[4*i+3]   += (two + mass + g * (*M)) * inVec[4*i+3];
    }
    }

}




template <typename T>
__global__ void D_eo(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int idx[2];
    int nt;

    T sgn[2];

    for (int i = grid.thread_rank(); i < vol/2; i += grid.size()) {

{
    int n = i;
    int alpha = 0;
    if (n >= Sizes[0]*Sizes[1]/2) {
        alpha = 1;
        n -= Sizes[0]*Sizes[1]/2;
    }
    idx[0] = n / (Sizes[1]/2);
    if (idx[0] % 2) idx[1] = 2*((n % (Sizes[1]/2))) + (1-alpha);
    else idx[1] = 2*((n % (Sizes[1]/2))) + alpha; 
}
    nt = idx[0];
    sgn[0] = (nt == (Sizes[0]-1)) ? -1.0 : 1.0;
    sgn[1] = (nt == 0) ? -1.0 : 1.0;

    thrust::complex<T> psisum[2], psidiff[2];

    T constexpr half {0.5};
    
    if (useDagger == MatrixType::Dagger) {
        
        psisum[0]  = inVec[4*IUP[i][1] + 0] + inVec[IUP[i][1] + 1];
        psisum[1]  = inVec[4*IUP[i][1] + 2] + inVec[IUP[i][1] + 3];
        psidiff[0] = inVec[4*IDN[i][1] + 0] - inVec[IDN[i][1] + 1];
        psidiff[1] = inVec[4*IDN[i][1] + 2] - inVec[IDN[i][1] + 3];

        outVec[4*i + 0] -=  sgn[0] * inVec[4*IUP[i][0] + 0] + half*psidiff[0] + half*psisum[0];
        outVec[4*i + 2] -=  sgn[0] * inVec[4*IUP[i][0] + 2] + half*psidiff[1] + half*psisum[1];
        outVec[4*i + 1] -=  sgn[1] * inVec[4*IDN[i][0] + 1] - half*psidiff[0] + half*psisum[0];
        outVec[4*i + 3] -=  sgn[1] * inVec[4*IDN[i][0] + 3] - half*psidiff[1] + half*psisum[1];

    } else {

        
        psisum[0]  = inVec[IDN[i][1] + 0] + inVec[IDN[i][1] + 1];
        psisum[1]  = inVec[IDN[i][1] + 2] + inVec[IDN[i][1] + 3];
        psidiff[0] = inVec[IUP[i][1] + 0] - inVec[IUP[i][1] + 1];
        psidiff[1] = inVec[IUP[i][1] + 2] - inVec[IUP[i][1] + 3];

        outVec[i + 0] -=  sgn[1] * inVec[4*IDN[i][0] + 0] + half*psisum[0] + half*psidiff[0];
        outVec[i + 1] -=  sgn[1] * inVec[4*IDN[i][0] + 2] + half*psisum[1] + half*psidiff[1];
        outVec[i + 2] -=  sgn[0] * inVec[4*IUP[i][0] + 1] + half*psisum[0] - half*psidiff[0];
        outVec[i + 3] -=  sgn[0] * inVec[4*IUP[i][0] + 3] + half*psisum[1] - half*psidiff[1];

    }

    }                            

}


template <typename T>
__global__ void D_oe(cp<T> *inVec, cp<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int idx[2];
    T const half {0.5};

    T sgn[2];
    int nt;

    for (int j = grid.thread_rank(); j < vol/2; j += grid.size()) {
    int i = j + vol/2;
    
{
    int n = i;
    int alpha = 0;
    if (n >= Sizes[0]*Sizes[1]/2) {
        alpha = 1;
        n -= Sizes[0]*Sizes[1]/2;
    }
    idx[0] = n / (Sizes[1]/2);
    if (idx[0] % 2) idx[1] = 2*((n % (Sizes[1]/2))) + (1-alpha);
    else idx[1] = 2*((n % (Sizes[1]/2))) + alpha; 
}

        nt = idx[0];

        sgn[0] = (nt == (Sizes[0]-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        thrust::complex<T> psisum[2], psidiff[2];
        
		i = j;
        if (useDagger == MatrixType::Dagger) {

            psisum[0]  = inVec[4*IUP[i+vol/2][1] + 0] + inVec[4*IUP[i+vol/2][1] + 1];
            psisum[1]  = inVec[4*IUP[i+vol/2][1] + 2] + inVec[4*IUP[i+vol/2][1] + 3];
            psidiff[0] = inVec[4*IDN[i+vol/2][1] + 0] - inVec[4*IDN[i+vol/2][1] + 1];
            psidiff[1] = inVec[4*IDN[i+vol/2][1] + 2] - inVec[4*IDN[i+vol/2][1] + 3];

            outVec[4*(i+vol/2) + 0] -=  sgn[0] * inVec[4*IUP[i+vol/2][0] + 0] + half*(psidiff[0] + psisum[0]);
            outVec[4*(i+vol/2) + 2] -=  sgn[0] * inVec[4*IUP[i+vol/2][0] + 2] + half*(psidiff[1] + psisum[1]);
            outVec[4*(i+vol/2) + 1] -=  sgn[1] * inVec[4*IDN[i+vol/2][0] + 1] - half*(psidiff[0] - psisum[0]);
            outVec[4*(i+vol/2) + 3] -=  sgn[1] * inVec[4*IDN[i+vol/2][0] + 3] - half*(psidiff[1] - psisum[1]);

        } else {

            psisum[0]  = inVec[4*IDN[i+vol/2][1] + 0] + inVec[4*IDN[i+vol/2][1] + 1];
            psisum[1]  = inVec[4*IDN[i+vol/2][1] + 2] + inVec[4*IDN[i+vol/2][1] + 3];
            psidiff[0] = inVec[4*IUP[i+vol/2][1] + 0] - inVec[4*IUP[i+vol/2][1] + 1];
            psidiff[1] = inVec[4*IUP[i+vol/2][1] + 2] - inVec[4*IUP[i+vol/2][1] + 3];

            outVec[4*(i+vol/2) + 0] -=  sgn[1] * inVec[4*IDN[i+vol/2][0] + 0] + half*psisum[0] + half*psidiff[0];
            outVec[4*(i+vol/2) + 2] -=  sgn[1] * inVec[4*IDN[i+vol/2][0] + 2] + half*psisum[1] + half*psidiff[1];
            outVec[4*(i+vol/2) + 1] -=  sgn[0] * inVec[4*IUP[i+vol/2][0] + 1] + half*psisum[0] - half*psidiff[0];
            outVec[4*(i+vol/2) + 3] -=  sgn[0] * inVec[4*IUP[i+vol/2][0] + 3] + half*psisum[1] - half*psidiff[1];


        }
    }
                                            
}


template class DiracOP<double>;

