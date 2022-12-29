#include "functions.h" 

void CG(SpinorField const& inPsi, SpinorField& outPsi, DiracOP Dirac, bool hermitian){
    assert(inPsi.Nt == outPsi.Nt && inPsi.Nx == outPsi.Nx && inPsi.Nf == outPsi.Nf);
    
    SpinorField 
        r(inPsi.Nt, inPsi.Nx, inPsi.Nf), // residual
        p(inPsi.Nt, inPsi.Nx, inPsi.Nf), 
        temp(inPsi.Nt, inPsi.Nx, inPsi.Nf); // Store results of D Ddagger psi = D _temp
    std::complex<double> alpha;
    double beta, rmodsq;
    
    for(auto& v: outPsi.val) v = 0.0;
    r.val = inPsi.val;
    p.val = r.val;
    rmodsq = (r.dot(r)).real();

    int k;
    for(k=0; k<IterMax && sqrt(rmodsq) > tol; k++){

        temp = Dirac.applyTo(Dirac.applyTo(p, 1));

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
    else std::cout << "Max. number of iterations reached (" << IterMax << ") \n";
}

unsigned int PBC(int const n, int const N){
    return (n+N) % N;
}

unsigned int toFlat(int const nt, int const nx, int const f, int const c) {
    return (c + 2*f + 2*nx*Nf + 2*nt*Nx*Nf);
}

unsigned int toEOflat(int const nt, int const nx, int const f, int const c) {
    int const s = (2*Nt*Nx*Nf)/2;
    int eo = (nt+nx) % 2;
    return c + 2*f + (nx/2)*2*Nf + (nt*(2*Nx/2*Nf)) + eo*s;
}

std::vector<int> toVec(int const idx){
    std::vector<int> r(4); // Nt, Nx, Nf, c
    r[0] = idx / (2*Nx*Nf);
    r[1] = (idx % (2*Nx*Nf)) / (2*Nf);
    r[2] = (idx % (2*Nf)) / 2;
    r[3] = idx % 2;
    return r;
}

std::vector<int> eoToVec(int n){
    std::vector<int> idx(4); // nt, nx, f, c
    int alpha = 0;
    if (n >= Nt*Nx*Nf) {
        alpha = 1;
        n -= Nt*Nx*Nf;
    }
    idx[3] = n % 2;
    idx[2] = (n % (2*Nf)) / 2;
    idx[0] = n / (2*Nf*Nx/2);
    if (idx[0] % 2) idx[1] = 2*((n % (2*Nf*Nx/2)) / (2*Nf)) + (1-alpha);
    else idx[1] = 2*((n % (2*Nf*Nx/2)) / (2*Nf)) + alpha; 
    return idx;
}

void MatMatprod(mat const& M1, mat const& M2, mat& res){
    for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
            res[i][j] = 0.0;
            for(int k=0; k<2; k++){
                res[i][j] += M1[i][k]*M2[k][j];
            }
        }
    }
}