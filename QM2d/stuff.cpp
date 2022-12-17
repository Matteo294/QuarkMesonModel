#include "stuff.h"

std::vector<int> toVec(int const idx){
    std::vector<int> r(4); // Nt, Nx, Nf, c
    r[0] = idx / (2*Nx*Nf);
    r[1] = (idx % (2*Nx*Nf)) / (2*Nf);
    r[2] = (idx % (2*Nf)) / 2;
    r[3] = idx % 2;
    return r;
}


void MatMatprod(std::vector<std::vector<std::complex<double>>> const& M1, std::vector<std::vector<std::complex<double>>> const& M2, std::vector<std::vector<std::complex<double>>>& res){
    for(int i=0; i<2; i++){
        for(int j=0; j<2; j++){
            res[i][j] = 0.0;
            for(int k=0; k<2; k++){
                res[i][j] += M1[i][k]*M2[k][j];
            }
        }
    }
}