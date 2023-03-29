#include "Dirac.cuh"


template class DiracOP<double>;


template <typename T>
__host__ void DiracOP<T>::applyD(void** diagArgs, void** hoppingArgs){

    int numBlocks = 0;
    int numThreads = 0;
    cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, D_eo<T>);

    auto dimGrid = dim3(numBlocks, 1, 1);
    auto dimBlock = dim3(numThreads, 1, 1);

    cudaLaunchCooperativeKernel((void*)&D_ee<T>, dimGrid, dimBlock, diagArgs, 0, NULL);
    cudaDeviceSynchronize();
    cudaLaunchCooperativeKernel((void*)&D_oo<T>, dimGrid, dimBlock, diagArgs, 0, NULL);
    cudaDeviceSynchronize();
    cudaLaunchCooperativeKernel((void*)&D_eo<T>, dimGrid, dimBlock, hoppingArgs, 0, NULL);
    cudaDeviceSynchronize();
    cudaLaunchCooperativeKernel((void*)&D_oe<T>, dimGrid, dimBlock, hoppingArgs, 0, NULL);
    cudaDeviceSynchronize();

}

template <typename T>
__global__ void D_oo(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, T const fermion_mass, T const g_coupling, MatrixType const useDagger, thrust::complex<T> *M){

    auto grid = cg::this_grid();

    thrust::complex<T> sigma;
    thrust::complex<T> const g = static_cast<thrust::complex<T>> (g_coupling);
    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> mass = static_cast<thrust::complex<T>> (fermion_mass);
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
__global__ void D_ee(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, T const fermion_mass, T const g_coupling, MatrixType const useDagger, thrust::complex<T> *M){

    auto grid = cg::this_grid();

    thrust::complex<T> sigma;
    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> const g = static_cast<thrust::complex<T>> (g_coupling);
    thrust::complex<T> const mass = static_cast<thrust::complex<T>> (fermion_mass);
    thrust::complex<T> const im {0.0, 1.0};
    thrust::complex<T> half {0.5, 0.0};

    thrust::complex<T> *M1 = M;
    thrust::complex<T> *M2 = &M[vol];
    thrust::complex<T> *M3 = &M[2*vol];
    thrust::complex<T> *M4 = &M[3*vol];

    for (int i = grid.thread_rank(); i < vol/2; i += grid.size()) {
    // Diagonal term

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
__global__ void D_ee_inv(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, T const fermion_mass, T const g_coupling, MatrixType const useDagger, thrust::complex<T> *M){
        auto grid = cg::this_grid();
    thrust::complex<T> sigma, det;
    thrust::complex<T> Minv[4];
    thrust::complex<T> const g = static_cast<thrust::complex<T>> (g_coupling)  ;
    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> const mass = static_cast<thrust::complex<T>> (fermion_mass);
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
__global__ void D_eo(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int idx[2];
    int nt;

    T sgn[2];

    for (int i = grid.thread_rank(); i < vol/2; i += grid.size()) {

{
    int n = i;
    int alpha = 0;
    if (n >= Nt*Nx/2) {
        alpha = 1;
        n -= Nt*Nx/2;
    }
    idx[0] = n / (Nx/2);
    if (idx[0] % 2) idx[1] = 2*((n % (Nx/2))) + (1-alpha);
    else idx[1] = 2*((n % (Nx/2))) + alpha; 
}
    nt = idx[0];
    sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
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
__global__ void D_oe(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

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
    if (n >= Nt*Nx/2) {
        alpha = 1;
        n -= Nt*Nx/2;
    }
    idx[0] = n / (Nx/2);
    if (idx[0] % 2) idx[1] = 2*((n % (Nx/2))) + (1-alpha);
    else idx[1] = 2*((n % (Nx/2))) + alpha; 
}

        nt = idx[0];

        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
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
__global__ void D_oo_inv(Spinor<T> *inVec, Spinor<T> *outVec, int const vol, T const fermion_mass, T const g_coupling, MatrixType const useDagger, thrust::complex<T> *M){

        auto grid = cg::this_grid();

    thrust::complex<T> sigma, det;
    thrust::complex<T> Minv[4];
    thrust::complex<T> const g = static_cast<thrust::complex<T>> (g_coupling);
    thrust::complex<T> const two {2.0, 0.0};
    thrust::complex<T> const mass = static_cast<thrust::complex<T>> (fermion_mass);
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



/*template <typename T>
void DiracOP<T>::applyDhat(Spinor<T> *inVec, Spinor<T> *outVec, MesonsMat<T> *M, MatrixType const useDagger){
    
    auto dimGrid = dim3(NBLOCKS, 1, 1);
    auto dimBlock = dim3(THREADS_PER_BLOCK, 1, 1);

    // check in all functions call whether we are working on the correct part of the array

    // control if we are cycling in the correct part!!!!!!!!!!!!!!
    for(int i=0; i<lattice.vol; i++) {
        outVec[i].setZero();
        temp[i].setZero();
        temp2[i].setZero();
    }

    // for the useDagger consider inverting the order of the product

    D_ee_wrap(inVec, outVec, M, useDagger);
    cudaDeviceSynchronize();

    D_oe_wrap(inVec, temp, M, useDagger);
    cudaDeviceSynchronize();

    D_oo_inv_wrap(temp, temp2, M, useDagger);
    cudaDeviceSynchronize();

    for(int i=0; i<lattice.vol; i++) {
        temp[i].setZero(); 
    }

    D_eo_wrap(temp2, temp, M, useDagger);
    cudaDeviceSynchronize();

    for(int i=0; i<lattice.vol/2; i++) { for (int j=0; j<4; j++) outVec[i].val[j] -= temp[i].val[j]; }

}*/
