#include "functions.h" 

void CG(SpinorField const& inPsi, SpinorField& outPsi, DiracOP Dirac, bool hermitian){
    assert(inPsi.Nt == outPsi.Nt && inPsi.Nx == outPsi.Nx && inPsi.Nf == outPsi.Nf);
    
    SpinorField 
        r(inPsi.Nt, inPsi.Nx, inPsi.Nf), // residual
        p(inPsi.Nt, inPsi.Nx, inPsi.Nf), 
        temp(inPsi.Nt, inPsi.Nx, inPsi.Nf); // Store results of D Ddagger psi = D _temp
    std::complex<double> alpha;
    double beta, rmodsq;
    
    for(auto& v: outPsi.val) v.setZero();
    r.val = inPsi.val;
    p.val = r.val;
    rmodsq = (r.dot(r)).real();

    int k;
    for(k=0; k<IterMax && sqrt(rmodsq) > tol; k++){

        temp = Dirac.applyToCSR(Dirac.applyToCSR(p, 1));

        alpha = rmodsq / (p.dot(temp)).real(); 
        for(int i=0; i<inPsi.volume; i++){
            outPsi.val[i] += alpha * p.val[i];
            r.val[i] -= alpha * temp.val[i]; 
        }
        beta = (r.dot(r)).real() / rmodsq;
        for(int i=0; i<inPsi.volume; i++) p.val[i] = r.val[i] + beta*p.val[i];
        rmodsq = (r.dot(r)).real();
    }

    if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
    else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << rmodsq << "\n";
}

unsigned int PBC(int const n, int const N){
    return (n+N) % N;
}

unsigned int toEOflat(int const nt, int const nx){
    int const s = (Nt*Nx)/2;
    int eo = (nt+nx) % 2;
    return (nx/2) + (nt*Nx/2) + eo*s;
}

std::vector<int> eoToVec(int n){
    std::vector<int> idx(2); // nt, nx
    int alpha = 0;
    if (n >= Nt*Nx/2) {
        alpha = 1;
        n -= Nt*Nx/2;
    }
    idx[0] = n / (Nx/2);
    if (idx[0] % 2) idx[1] = 2*((n % (Nx/2))) + (1-alpha);
    else idx[1] = 2*((n % (Nx/2))) + alpha; 
    return idx;
}

mat_fc buildCompositeOP(mat const& flavourMat, mat const& spinorMat){
    mat_fc M;
    for(int f1=0; f1<Nf; f1++){
        for(int f2=0; f2<Nf; f2++){
            for(int c1=0; c1<2; c1++){
                for(int c2=0; c2<2; c2++){
                    M(2*f1+c1, 2*f2+c2) = flavourMat(f1, f2) * spinorMat(c1, c2);
                }
            }
        }
    }
    return M;
}
