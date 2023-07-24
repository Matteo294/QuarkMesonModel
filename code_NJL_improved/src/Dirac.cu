#include "Dirac.cuh"

extern __constant__ double yukawa_coupling_gpu;
extern __constant__ double fermion_mass_gpu;
extern __constant__ thrust::complex<double> im_gpu;


template <typename T>
__host__ DiracOP<T>::DiracOP() : inVec(nullptr), outVec(nullptr), M(nullptr)
	{

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
        diagArgs[4] = (void*) &EO2N.at;
		hoppingArgs[0] = (void*) &inVec;
		hoppingArgs[1] = (void*) &outVec;
		hoppingArgs[2] = (void*) &useDagger;
		hoppingArgs[3] = (void*) &IUP.at;
		hoppingArgs[4] = (void*) &IDN.at;
        
        //for(int i=0; i<vol; i++) EO2N.at[i] = convertEOtoNormal(i);

    }



__device__ void D_oo(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M, int *EO2N){

    auto grid = cg::this_grid();

    thrust::complex<double> const g = static_cast<thrust::complex<double>> (yukawa_coupling_gpu);
    thrust::complex<double> const two {2.0, 0.0};
    thrust::complex<double> mass = static_cast<thrust::complex<double>> (fermion_mass_gpu);
    thrust::complex<double> half {0.5, 0.0};

    int Ni;
    for (int i = grid.thread_rank() + vol/2; i < vol; i += grid.size()) {
        Ni = EO2N[i];
        if (useDagger == MatrixType::Dagger){
            outVec[4*i]     += (two + mass + g * M[Ni]) * inVec[4*i];
            outVec[4*i+1]   += (two + mass + g * M[Ni]) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * M[Ni]) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * M[Ni]) * inVec[4*i+3];
        } else{
            outVec[4*i]     += (two + mass + g * (M[Ni])) * inVec[4*i];
            outVec[4*i+1]   += (two + mass + g * (M[Ni])) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * (M[Ni])) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * (M[Ni])) * inVec[4*i+3];
        }
    }
}

__device__ void D_ee(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M, int *EO2N){

    auto grid = cg::this_grid();

    thrust::complex<double> const two {2.0, 0.0};
    thrust::complex<double> const g = static_cast<thrust::complex<double>> (yukawa_coupling_gpu);
    thrust::complex<double> const mass = static_cast<thrust::complex<double>> (fermion_mass_gpu);
    thrust::complex<double> half {0.5, 0.0};

    int Ni;
    for (int i = grid.thread_rank(); i < vol/2; i += grid.size()) {
        Ni = EO2N[i];
        if (useDagger == MatrixType::Dagger){
            outVec[4*i]     += (two + mass + g * M[Ni]) * inVec[4*i];
            outVec[4*i+1]   += (two + mass + g * M[Ni]) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * M[Ni]) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * M[Ni]) * inVec[4*i+3];
        } else {
            outVec[4*i]     += (two + mass + g * (M[Ni])) * inVec[4*i];
            outVec[4*i+1]   += (two + mass + g * (M[Ni])) * inVec[4*i+1];
            outVec[4*i+2]   += (two + mass + g * (M[Ni])) * inVec[4*i+2];
            outVec[4*i+3]   += (two + mass + g * (M[Ni])) * inVec[4*i+3];
        }
    }

}




__device__ void D_eo(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int idx[2];
    int nt;

    double sgn[2];

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

    thrust::complex<double> psisum[2], psidiff[2];

    double constexpr half {0.5};
    
    if (useDagger == MatrixType::Dagger) {
        
        psisum[0]  = inVec[4*IUP[i][1] + 0] + inVec[4*IUP[i][1] + 1];
        psisum[1]  = inVec[4*IUP[i][1] + 2] + inVec[4*IUP[i][1] + 3];
        psidiff[0] = inVec[4*IDN[i][1] + 0] - inVec[4*IDN[i][1] + 1];
        psidiff[1] = inVec[4*IDN[i][1] + 2] - inVec[4*IDN[i][1] + 3];

        outVec[4*i + 0] -=  sgn[0] * inVec[4*IUP[i][0] + 0] + half*psidiff[0] + half*psisum[0];
        outVec[4*i + 2] -=  sgn[0] * inVec[4*IUP[i][0] + 2] + half*psidiff[1] + half*psisum[1];
        outVec[4*i + 1] -=  sgn[1] * inVec[4*IDN[i][0] + 1] - half*psidiff[0] + half*psisum[0];
        outVec[4*i + 3] -=  sgn[1] * inVec[4*IDN[i][0] + 3] - half*psidiff[1] + half*psisum[1];

    } else {

        
        psisum[0]  = inVec[4*IDN[i][1] + 0] + inVec[4*IDN[i][1] + 1];
        psisum[1]  = inVec[4*IDN[i][1] + 2] + inVec[4*IDN[i][1] + 3];
        psidiff[0] = inVec[4*IUP[i][1] + 0] - inVec[4*IUP[i][1] + 1];
        psidiff[1] = inVec[4*IUP[i][1] + 2] - inVec[4*IUP[i][1] + 3];

        outVec[4*i + 0] -=  sgn[1] * inVec[4*IDN[i][0] + 0] + half*psisum[0] + half*psidiff[0];
        outVec[4*i + 2] -=  sgn[1] * inVec[4*IDN[i][0] + 2] + half*psisum[1] + half*psidiff[1];
        outVec[4*i + 1] -=  sgn[0] * inVec[4*IUP[i][0] + 1] + half*psisum[0] - half*psidiff[0];
        outVec[4*i + 3] -=  sgn[0] * inVec[4*IUP[i][0] + 3] + half*psisum[1] - half*psidiff[1];

    }

    }                            

}


__device__ void D_oe(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, my2dArray *IUP, my2dArray *IDN){

    auto grid = cg::this_grid();

    int idx[2];
    double const half {0.5};

    double sgn[2];
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

        thrust::complex<double> psisum[2], psidiff[2];
        
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


//template<> void D_oo<double>(cp<double> *inVec, cp<double> *outVec, MatrixType const useDagger, double *M, int *EO2N);

template class DiracOP<double>;

