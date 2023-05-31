#include "Dirac.cuh"

extern __constant__ double yukawa_coupling_gpu;
extern __constant__ double fermion_mass_gpu;

template <typename T>
__host__ DiracOP<T>::DiracOP() : inVec(nullptr), outVec(nullptr), M(nullptr)
	{
        cudaMallocManaged(&temp, sizeof(Spinor<T>) * vol/2); 
        cudaMallocManaged(&temp2, sizeof(Spinor<T>) * vol/2);
        
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

    for(int i=0; i<vol; i++) outVec[i].setZero(); // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    cudaLaunchCooperativeKernel((void*)&D_ee<T>, dimGrid_Dee, dimBlock_Dee, diagArgs, 0, NULL);
    cudaDeviceSynchronize();
    std::cout << "Dee " << outVec[0].val[0] << " " << outVec[vol/2].val[4] << std::endl;

    cudaLaunchCooperativeKernel((void*)&D_oo<T>, dimGrid_Doo, dimBlock_Doo, diagArgs, 0, NULL);
    cudaDeviceSynchronize();
    std::cout << "Dee " << outVec[0].val[0] << " " << outVec[vol/2].val[4] << std::endl;

   	cudaLaunchCooperativeKernel((void*)&D_eo<T>, dimGrid_Deo, dimBlock_Deo, hoppingArgs, 0, NULL);
    cudaDeviceSynchronize();

    cudaLaunchCooperativeKernel((void*)&D_oe<T>, dimGrid_Doe, dimBlock_Doe, hoppingArgs, 0, NULL);
    cudaDeviceSynchronize();

}


template <typename T>
__global__ void D_oo(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, thrust::complex<T> *M){

    auto grid = cg::this_grid();


    thrust::complex<T> sigma;
    thrust::complex<T> const g = static_cast<thrust::complex<T>> (yukawa_coupling_gpu);
    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> mass = static_cast<thrust::complex<T>> (fermion_mass_gpu);
    thrust::complex<T> const im {0.0, 1.0};
    thrust::complex<T> half {0.5, 0.0};

    thrust::complex<T> *M1 = M;
    thrust::complex<T> *M2 = &M[vol];
    thrust::complex<T> *M3 = &M[2*vol];
    thrust::complex<T> *M4 = &M[3*vol];

    for (int i = grid.thread_rank() + vol/2; i < vol; i += grid.size()) {
        sigma = half * (M1[i] + M4[i]);
        if (useDagger == MatrixType::Dagger){
            outVec[i].val[0] +=  im * g * ( conj((M1[i] - sigma)) * inVec[i].val[1] +         conj(M3[i])     * inVec[i].val[3] ) + (two + mass + g*conj(sigma)) * inVec[i].val[0];
            outVec[i].val[1] += -im * g * ( conj((M1[i] - sigma)) * inVec[i].val[0] +         conj(M3[i])     * inVec[i].val[2] ) + (two + mass + g*conj(sigma)) * inVec[i].val[1];
            outVec[i].val[2] +=  im * g * (       conj(M2[i])     * inVec[i].val[1] + conj((M4[i] - sigma))   * inVec[i].val[3] ) + (two + mass + g*conj(sigma)) * inVec[i].val[2];
            outVec[i].val[3] += -im * g * (       conj(M2[i])     * inVec[i].val[0] + conj((M4[i] - sigma))   * inVec[i].val[2] ) + (two + mass + g*conj(sigma)) * inVec[i].val[3];
        } else{
            outVec[i].val[0] +=  im * g * ( (M1[i] - sigma) * inVec[i].val[1] +         M2[i]     * inVec[i].val[3] ) + (two + mass + g*sigma) * inVec[i].val[0];
            outVec[i].val[1] += -im * g * ( (M1[i] - sigma) * inVec[i].val[0] +         M2[i]     * inVec[i].val[2] ) + (two + mass + g*sigma) * inVec[i].val[1];
            outVec[i].val[2] +=  im * g * (       M3[i]     * inVec[i].val[1] + (M4[i] - sigma)   * inVec[i].val[3] ) + (two + mass + g*sigma) * inVec[i].val[2];
            outVec[i].val[3] += -im * g * (       M3[i]     * inVec[i].val[0] + (M4[i] - sigma)   * inVec[i].val[2] ) + (two + mass + g*sigma) * inVec[i].val[3];
        }
    }
}

template <typename T>
__global__ void D_ee(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, thrust::complex<T> *M){

    auto grid = cg::this_grid();

    thrust::complex<T> sigma;
    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> const g = static_cast<thrust::complex<T>> (yukawa_coupling_gpu);
    thrust::complex<T> const mass = static_cast<thrust::complex<T>> (fermion_mass_gpu);
    thrust::complex<T> const im {0.0, 1.0};
    thrust::complex<T> half {0.5, 0.0};

    thrust::complex<T> *M1 = M;
    thrust::complex<T> *M2 = &M[vol];
    thrust::complex<T> *M3 = &M[2*vol];
    thrust::complex<T> *M4 = &M[3*vol];

    for (int i = grid.thread_rank(); i < vol/2; i += grid.size()) {
    


    sigma = half * (M1[i] + M4[i]);        
    if (useDagger == MatrixType::Dagger){
        outVec[i].val[0] +=  im * g * ( conj((M1[i] - sigma)) * inVec[i].val[1] +         conj(M3[i])     * inVec[i].val[3] ) + (two + mass + g*conj(sigma)) * inVec[i].val[0];
        outVec[i].val[1] += -im * g * ( conj((M1[i] - sigma)) * inVec[i].val[0] +         conj(M3[i])     * inVec[i].val[2] ) + (two + mass + g*conj(sigma)) * inVec[i].val[1];
        outVec[i].val[2] +=  im * g * (       conj(M2[i])     * inVec[i].val[1] + conj((M4[i] - sigma))   * inVec[i].val[3] ) + (two + mass + g*conj(sigma)) * inVec[i].val[2];
        outVec[i].val[3] += -im * g * (       conj(M2[i])     * inVec[i].val[0] + conj((M4[i] - sigma))   * inVec[i].val[2] ) + (two + mass + g*conj(sigma)) * inVec[i].val[3];
    } else {
        outVec[i].val[0] +=  im * g * ( (M1[i] - sigma) * inVec[i].val[1] +         M2[i]     * inVec[i].val[3] ) + (two + mass + g*sigma) * inVec[i].val[0];
        outVec[i].val[1] += -im * g * ( (M1[i] - sigma) * inVec[i].val[0] +         M2[i]     * inVec[i].val[2] ) + (two + mass + g*sigma) * inVec[i].val[1];
        outVec[i].val[2] +=  im * g * (       M3[i]     * inVec[i].val[1] + (M4[i] - sigma)   * inVec[i].val[3] ) + (two + mass + g*sigma) * inVec[i].val[2];
        outVec[i].val[3] += -im * g * (       M3[i]     * inVec[i].val[0] + (M4[i] - sigma)   * inVec[i].val[2] ) + (two + mass + g*sigma) * inVec[i].val[3];
    }
    }

}


template <typename T>
__global__ void D_ee_inv(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, thrust::complex<T> *M){
        auto grid = cg::this_grid();

    thrust::complex<T> sigma, det;
    thrust::complex<T> Minv[4];
    thrust::complex<T> const g = static_cast<thrust::complex<T>> (yukawa_coupling_gpu)  ;
    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> const mass = static_cast<thrust::complex<T>> (fermion_mass_gpu);
    thrust::complex<T> const im {0.0, 1.0};

    thrust::complex<T> *M1 = M;
    thrust::complex<T> *M2 = &M[vol];
    thrust::complex<T> *M3 = &M[2*vol];
    thrust::complex<T> *M4 = &M[3*vol];

    for (int i = grid.thread_rank(); i < vol/2; i += grid.size()) {

    Minv[0] = g*M4[i] + two + mass;
    Minv[3] = g*M1[i] + two + mass;
    Minv[1] = - g*M2[i];
    Minv[2] = - g*M3[i];
    det = Minv[0]*Minv[3] - Minv[1]*Minv[2];
    Minv[0] /= det;
    Minv[1] /= det; 
    Minv[2] /= det; 
    Minv[3] /= det; 

    sigma = (thrust::complex<T>) 0.5 * (Minv[0] + Minv[3]); // this is actually g sigma + m + 2

    if (useDagger == MatrixType::Dagger){
        outVec[i].val[0] +=  im * ( conj((Minv[0] - sigma))    * inVec[i].val[1] +         conj(Minv[2])     * inVec[i].val[3] ) + sigma * inVec[i].val[0];
        outVec[i].val[1] += -im * ( conj((Minv[0] - sigma))    * inVec[i].val[0] +         conj(Minv[2])     * inVec[i].val[2] ) + sigma * inVec[i].val[1];
        outVec[i].val[2] +=  im * (          conj(Minv[1])     * inVec[i].val[1] + conj((Minv[3] - sigma))   * inVec[i].val[3] ) + sigma * inVec[i].val[2];
        outVec[i].val[3] += -im * (          conj(Minv[1])     * inVec[i].val[0] + conj((Minv[3] - sigma))   * inVec[i].val[2] ) + sigma * inVec[i].val[3];
    } else {
        outVec[i].val[0] +=  im * ( (Minv[0] - sigma) * inVec[i].val[1] +      Minv[1]       * inVec[i].val[3] ) + sigma * inVec[i].val[0];
        outVec[i].val[1] += -im * ( (Minv[0] - sigma) * inVec[i].val[0] +      Minv[1]       * inVec[i].val[2] ) + sigma * inVec[i].val[1];
        outVec[i].val[2] +=  im * (       Minv[2]     * inVec[i].val[1] + (Minv[3] - sigma)  * inVec[i].val[3] ) + sigma * inVec[i].val[2];
        outVec[i].val[3] += -im * (       Minv[2]     * inVec[i].val[0] + (Minv[3] - sigma)  * inVec[i].val[2] ) + sigma * inVec[i].val[3];
    }
    }
}


template <typename T>
__global__ void D_eo(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

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
        
        psisum[0]  = inVec[IUP[i][1]].val[0] + inVec[IUP[i][1]].val[1];
        psisum[1]  = inVec[IUP[i][1]].val[2] + inVec[IUP[i][1]].val[3];
        psidiff[0] = inVec[IDN[i][1]].val[0] - inVec[IDN[i][1]].val[1];
        psidiff[1] = inVec[IDN[i][1]].val[2] - inVec[IDN[i][1]].val[3];

        outVec[i].val[0] -=  sgn[0] * inVec[IUP[i][0]].val[0] + half*psidiff[0] + half*psisum[0];
        outVec[i].val[2] -=  sgn[0] * inVec[IUP[i][0]].val[2] + half*psidiff[1] + half*psisum[1];
        outVec[i].val[1] -=  sgn[1] * inVec[IDN[i][0]].val[1] - half*psidiff[0] + half*psisum[0];
        outVec[i].val[3] -=  sgn[1] * inVec[IDN[i][0]].val[3] - half*psidiff[1] + half*psisum[1];

    } else {

        
        psisum[0]  = inVec[IDN[i][1]].val[0] + inVec[IDN[i][1]].val[1];
        psisum[1]  = inVec[IDN[i][1]].val[2] + inVec[IDN[i][1]].val[3];
        psidiff[0] = inVec[IUP[i][1]].val[0] - inVec[IUP[i][1]].val[1];
        psidiff[1] = inVec[IUP[i][1]].val[2] - inVec[IUP[i][1]].val[3];

        outVec[i].val[0] -=  sgn[1] * inVec[IDN[i][0]].val[0] + half*psisum[0] + half*psidiff[0];
        outVec[i].val[2] -=  sgn[1] * inVec[IDN[i][0]].val[2] + half*psisum[1] + half*psidiff[1];
        outVec[i].val[1] -=  sgn[0] * inVec[IUP[i][0]].val[1] + half*psisum[0] - half*psidiff[0];
        outVec[i].val[3] -=  sgn[0] * inVec[IUP[i][0]].val[3] + half*psisum[1] - half*psidiff[1];

    }

    }                            

}


template <typename T>
__global__ void D_oe(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

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

            psisum[0]  = inVec[IUP[i+vol/2][1]].val[0] + inVec[IUP[i+vol/2][1]].val[1];
            psisum[1]  = inVec[IUP[i+vol/2][1]].val[2] + inVec[IUP[i+vol/2][1]].val[3];
            psidiff[0] = inVec[IDN[i+vol/2][1]].val[0] - inVec[IDN[i+vol/2][1]].val[1];
            psidiff[1] = inVec[IDN[i+vol/2][1]].val[2] - inVec[IDN[i+vol/2][1]].val[3];

            outVec[i+vol/2].val[0] -=  sgn[0] * inVec[IUP[i+vol/2][0]].val[0] + half*(psidiff[0] + psisum[0]);
            outVec[i+vol/2].val[2] -=  sgn[0] * inVec[IUP[i+vol/2][0]].val[2] + half*(psidiff[1] + psisum[1]);
            outVec[i+vol/2].val[1] -=  sgn[1] * inVec[IDN[i+vol/2][0]].val[1] - half*(psidiff[0] - psisum[0]);
            outVec[i+vol/2].val[3] -=  sgn[1] * inVec[IDN[i+vol/2][0]].val[3] - half*(psidiff[1] - psisum[1]);

        } else {

            psisum[0]  = inVec[IDN[i+vol/2][1]].val[0] + inVec[IDN[i+vol/2][1]].val[1];
            psisum[1]  = inVec[IDN[i+vol/2][1]].val[2] + inVec[IDN[i+vol/2][1]].val[3];
            psidiff[0] = inVec[IUP[i+vol/2][1]].val[0] - inVec[IUP[i+vol/2][1]].val[1];
            psidiff[1] = inVec[IUP[i+vol/2][1]].val[2] - inVec[IUP[i+vol/2][1]].val[3];

            outVec[i+vol/2].val[0] -=  sgn[1] * inVec[IDN[i+vol/2][0]].val[0] + half*psisum[0] + half*psidiff[0];
            outVec[i+vol/2].val[2] -=  sgn[1] * inVec[IDN[i+vol/2][0]].val[2] + half*psisum[1] + half*psidiff[1];
            outVec[i+vol/2].val[1] -=  sgn[0] * inVec[IUP[i+vol/2][0]].val[1] + half*psisum[0] - half*psidiff[0];
            outVec[i+vol/2].val[3] -=  sgn[0] * inVec[IUP[i+vol/2][0]].val[3] + half*psisum[1] - half*psidiff[1];


        }
    }
                                            
}


template <typename T>
__global__ void D_oo_inv(Spinor<T> *inVec, Spinor<T> *outVec, MatrixType const useDagger, thrust::complex<T> *M){

        auto grid = cg::this_grid();

    thrust::complex<T> sigma, det;
    thrust::complex<T> Minv[4];
    thrust::complex<T> const g = static_cast<thrust::complex<T>> (yukawa_coupling_gpu);
    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> const mass = static_cast<thrust::complex<T>> (fermion_mass_gpu);
    thrust::complex<T> const im {0.0, 1.0};
    thrust::complex<T> half {0.5, 0.0};

    thrust::complex<T> *M1 = M;
    thrust::complex<T> *M2 = &M[vol];
    thrust::complex<T> *M3 = &M[2*vol];
    thrust::complex<T> *M4 = &M[3*vol];

    for (int i = grid.thread_rank() + vol/2; i < vol; i += grid.size()) {

        
    Minv[0] = g*M4[i] + two + mass;
    Minv[3] = g*M1[i] + two + mass;
    Minv[1] = - g*M2[i];
    Minv[2] = - g*M3[i];
    det = Minv[0]*Minv[3] - Minv[1]*Minv[2];
    Minv[0] /= det;
    Minv[1] /= det; 
    Minv[2] /= det; 
    Minv[3] /= det; 

    sigma = half * (Minv[0] + Minv[3]); // this is actually sigma + m + 2

    if (useDagger == MatrixType::Dagger){
        outVec[i].val[0] +=  im * ( conj((Minv[0] - sigma))    * inVec[i].val[1] +         conj(Minv[2])     * inVec[i].val[3] ) + sigma * inVec[i].val[0];
        outVec[i].val[1] += -im * ( conj((Minv[0] - sigma))    * inVec[i].val[0] +         conj(Minv[2])     * inVec[i].val[2] ) + sigma * inVec[i].val[1];
        outVec[i].val[2] +=  im * (          conj(Minv[1])     * inVec[i].val[1] + conj((Minv[3] - sigma))   * inVec[i].val[3] ) + sigma * inVec[i].val[2];
        outVec[i].val[3] += -im * (          conj(Minv[1])     * inVec[i].val[0] + conj((Minv[3] - sigma))   * inVec[i].val[2] ) + sigma * inVec[i].val[3];
    } else {
        outVec[i].val[0] +=  im * ( (Minv[0] - sigma) * inVec[i].val[1] +      Minv[1]       * inVec[i].val[3] ) + sigma * inVec[i].val[0];
        outVec[i].val[1] += -im * ( (Minv[0] - sigma) * inVec[i].val[0] +      Minv[1]       * inVec[i].val[2] ) + sigma * inVec[i].val[1];
        outVec[i].val[2] +=  im * (       Minv[2]     * inVec[i].val[1] + (Minv[3] - sigma)  * inVec[i].val[3] ) + sigma * inVec[i].val[2];
        outVec[i].val[3] += -im * (       Minv[2]     * inVec[i].val[0] + (Minv[3] - sigma)  * inVec[i].val[2] ) + sigma * inVec[i].val[3];
    }
    }

}


template class DiracOP<double>;

