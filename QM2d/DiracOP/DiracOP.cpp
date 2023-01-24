#include "DiracOP.h"

DiracOP::DiracOP(double const M, O4Mesons* mesons) : 
    M{M}, 
    mesons{mesons},
    GammaMat { mat {{2, 0}, {0, 0}},
            mat {{0, 0}, {0, 2}},
            mat {{1, 1}, {1, 1}}, 
            mat {{1, -1}, {-1, 1}} }
    {;}



SpinorField DiracOP::applyToCSR(SpinorField const& inPsi, bool const dagger){
    assert(dagger==1 or dagger==0);
    
    SpinorField outPsi(inPsi.Nt, inPsi.Nx, inPsi.Nf);

    for(int i=0; i<outPsi.volume; i++) outPsi.val[i].setZero();


    std::complex<double> sigma;
    std::vector<int> idx(2);

    // Diagonal term
    for(int i=0; i<inPsi.volume; i++){
        idx = eoToVec(i);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace();
        outPsi.val[i] += buildCompositeOP(mat::Identity(), Diag + mesons->g*sigma*mat::Identity()) * inPsi.val[i]; // flavour diagonal term
        if (dagger)
            outPsi.val[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()).adjoint(), gamma5) * inPsi.val[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            outPsi.val[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()), gamma5) * inPsi.val[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)

    }

    // Hopping term
    int const Nt=inPsi.Nt, Nx=inPsi.Nx;
    auto outPsi_copy = outPsi.val;

    // If dagger take adjoint projectors
    std::vector<mat> _Gamma;
    for(mat const& m: GammaMat) _Gamma.push_back(dagger ? gamma5*m*gamma5 : m);


    int const NNZ = 4*Nt*Nx; // 4 non-zero elements for each spacetime points (neighbours)
    std::vector<mat_fc> v;
    for(mat& m: _Gamma) v.push_back(buildCompositeOP(mat::Identity(), m)); // build composite op (i.e. add flavour indices)


    // Matrix-vector product in CSR format
    int COL_IDX[NNZ], ROW_IDX[Nt*Nx+1];
    int nt, nx;
    
    for(int i=0; i<NNZ; i+=4){
        nt = (i/(4*Nx)) % Nt;
        nx = (i/4) % Nx;
        COL_IDX[i]   = toEOflat(PBC(nt+1, Nt), nx);
        COL_IDX[i+1] = toEOflat(PBC(nt-1, Nt), nx);
        COL_IDX[i+2] = toEOflat(nt, PBC(nx+1, Nx));
        COL_IDX[i+3] = toEOflat(nt, PBC(nx-1, Nx));
        ROW_IDX[i/4] = i;
    }
    ROW_IDX[Nt*Nx] = NNZ;

    double sgn[2];
    for(int i=0; i<Nt*Nx; i++){
        nt = (i/Nx) % Nt;
        nx = (i) % Nx;
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;
        for(int j=ROW_IDX[i]; j<ROW_IDX[i+1]; j++){
            if (eoToVec(COL_IDX[j])[0] == (PBC(nt+1, Nt))) outPsi.val[toEOflat(nt, nx)] -= 0.5 * v[j%4] * sgn[0] * inPsi.val[COL_IDX[j]];
            else if (eoToVec(COL_IDX[j])[0] == (PBC(nt-1, Nt))) outPsi.val[toEOflat(nt, nx)] -= 0.5 * v[j%4] * sgn[1] * inPsi.val[COL_IDX[j]];
            else outPsi.val[toEOflat(nt, nx)] -= 0.5 * v[j%4] * inPsi.val[COL_IDX[j]];
        }
    }


    return outPsi;

}