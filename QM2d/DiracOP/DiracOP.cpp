#include "DiracOP.h"

// maybe bring setZero() inside Dee Doo, ...

DiracOP::DiracOP(double const M, O4Mesons* mesons, Lattice& l) : 
    M{M}, 
    mesons{mesons},
    l{l}
    {;}

void DiracOP::applyTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res, bool const dagger){
    assert(dagger==1 || dagger==0);
    for(int i=0; i<l.vol; i++) {res[i].setZero();}
    D_ee(vec, res, dagger);
    D_oo(vec + l.vol/2, res + l.vol/2, dagger);
    D_eo(vec + l.vol/2, res, dagger);
    D_oe(vec , res + l.vol/2, dagger);
}

void DiracOP::applyTo_single(std::vector<vec_fc_single>::iterator vec, std::vector<vec_fc_single>::iterator res, bool const dagger){
    assert(dagger==1 || dagger==0);
    for(int i=0; i<l.vol; i++) {res[i].setZero();}
    D_ee_single(vec, res, dagger);
    D_oo_single(vec + l.vol/2, res + l.vol/2, dagger);
    D_eo_single(vec + l.vol/2, res, dagger);
    D_oe_single(vec , res + l.vol/2, dagger);
}


void DiracOP::applyDhatTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res, bool const dagger){
    assert(dagger==0 || dagger==1);
    
    std::vector<vec_fc> temp(l.vol/2), temp2(l.vol/2);

    for(int i=0; i<l.vol/2; i++) {temp[i].setZero(); temp2[i].setZero(); res[i].setZero();}

    // for the dagger consider inverting the order of the product
    D_ee(vec, res, dagger);
    D_oe(vec, temp.begin(), dagger);  
    D_oo_inv(temp.begin(), temp2.begin(), dagger);
    for(int i=0; i<l.vol/2; i++) {temp[i].setZero();}
    D_eo(temp2.begin(), temp.begin(), dagger);

    for(int i=0; i<l.vol/2; i++) res[i] -= temp[i];
}

void DiracOP::applyDhatTo_single(std::vector<vec_fc_single>::iterator vec, std::vector<vec_fc_single>::iterator res, bool const dagger){
    assert(dagger==0 || dagger==1);
    
    std::vector<vec_fc_single> temp(l.vol/2), temp2(l.vol/2);

    for(int i=0; i<l.vol/2; i++) {temp[i].setZero(); temp2[i].setZero(); res[i].setZero();}

    // for the dagger consider inverting the order of the product
    D_ee_single(vec, res, dagger);
    D_oe_single(vec, temp.begin(), dagger);  
    D_oo_inv_single(temp.begin(), temp2.begin(), dagger);
    for(int i=0; i<l.vol/2; i++) {temp[i].setZero();}
    D_eo_single(temp2.begin(), temp.begin(), dagger);

    for(int i=0; i<l.vol/2; i++) res[i] -= temp[i];
}


void DiracOP::D_oo_inv(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger){
    std::complex<double> sigma;
    std::vector<int> idx(2);
    mat_fc Y;
    std::vector<std::complex<double>> pions(3);
    mat Mbar = mesons->M[0][0];
    mat_fc Minv;

    int const vol = l.vol/2;

    for(int i=0; i<vol; i++){
        idx = l.eoToVec(i+vol);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace();

        // Projection along basis
        pions[0] = 0.5 * ((Pauli.tau1*mesons->M[idx[0]][idx[1]]).trace());
        pions[1] = 0.5 * ((Pauli.tau2*mesons->M[idx[0]][idx[1]]).trace());
        pions[2] = 0.5 * ((Pauli.tau3*mesons->M[idx[0]][idx[1]]).trace());
        Mbar = - mesons->g*pions[0]*Pauli.tau1 - mesons->g*pions[1]*Pauli.tau2 - mesons->g*pions[2]*Pauli.tau3;
        
        if (dagger)
            Minv = buildCompositeOP(Mbar.adjoint(), gamma5); // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            Minv = buildCompositeOP(Mbar, gamma5); // flavour mixing term (just remove diagonal and tensor with gamma5)

        Minv += buildCompositeOP((2.0 + M + mesons->g*sigma)*Pauli.tau0, mat::Identity());
        Minv /= (2.0 + M + mesons->g*sigma)*(2.0 + M + mesons->g*sigma) - mesons->g*mesons->g*pions[0]*pions[0] - mesons->g*mesons->g*pions[1]*pions[1] - mesons->g*mesons->g*pions[2]*pions[2]; // - because the projection contains the i factor

        x[i] = Minv * y[i];

    }
}

// this must be improved
void DiracOP::D_oo_inv_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger){
    std::complex<float> sigma;
    std::vector<int> idx(2);
    mat_fc Y;
    std::vector<std::complex<float>> pions(3);
    mat Mbar = mesons->M[0][0];
    mat_fc Minv;

    int const vol = l.vol/2;

    for(int i=0; i<vol; i++){
        idx = l.eoToVec(i+vol);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace();

        // Projection along basis
        pions[0] = 0.5 * ((Pauli.tau1*mesons->M[idx[0]][idx[1]]).trace());
        pions[1] = 0.5 * ((Pauli.tau2*mesons->M[idx[0]][idx[1]]).trace());
        pions[2] = 0.5 * ((Pauli.tau3*mesons->M[idx[0]][idx[1]]).trace());
        Mbar = - (std::complex<float>) mesons->g*pions[0]*Pauli.tau1.cast<std::complex<float>>() - (std::complex<float>) mesons->g*pions[1]*Pauli.tau2.cast<std::complex<float>>() - (std::complex<float>) mesons->g*pions[2]*Pauli.tau3.cast<std::complex<float>>();
        
        if (dagger)
            Minv = buildCompositeOP(Mbar.adjoint(), gamma5); // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            Minv = buildCompositeOP(Mbar, gamma5); // flavour mixing term (just remove diagonal and tensor with gamma5)

        Minv += buildCompositeOP((2.0 + (float)M + (std::complex<float>)mesons->g*sigma)*Pauli.tau0, mat::Identity());
        Minv /= (2.0 + (float)M + (std::complex<float>)mesons->g*sigma)*(2.0 + M + (std::complex<float>)mesons->g*sigma) - (std::complex<float>)mesons->g*(std::complex<float>)mesons->g*pions[0]*pions[0] - (std::complex<float>)mesons->g*(std::complex<float>)mesons->g*pions[1]*pions[1] - (std::complex<float>)mesons->g*(std::complex<float>)mesons->g*pions[2]*pions[2]; // - because the projection contains the i factor

        x[i] = Minv * y[i];

    }
}

void DiracOP::D_ee(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger){

    std::complex<double> sigma;
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        idx = l.eoToVec(i);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace();
        x[i] += (2.0 + M + mesons->g*sigma) * y[i];
        if (dagger)
            x[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()).adjoint(), gamma5) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            x[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()), gamma5) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)

    }

}

void DiracOP::D_oo(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger){
    std::complex<double> sigma;
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        idx = l.eoToVec(i+vol);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace();
        x[i] += (2.0 + M + mesons->g*sigma) * y[i];
        if (dagger)
            x[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()).adjoint(), gamma5) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            x[i] += buildCompositeOP(mesons->g * (mesons->M[idx[0]][idx[1]] - sigma*mat::Identity()), gamma5) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)

    }
}


void DiracOP::D_eo(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger){

    std::vector<int> idx(2);
    int const vol = l.vol/2;


    int const Nt=l.Nt, Nx=l.Nx;

    double sgn[2];
    int nt, nx;
    
    for(int i=0; i<vol; i++){

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<double> psisum[2], psidiff[2];

        // maybe use ? : syntax to remove the if (more elegant?)
        
        if (dagger) {

            psisum[0] = y[l.IDN[i][1] - vol][0] + y[l.IDN[i][1] - vol][1];
            psisum[1] = y[l.IDN[i][1] - vol][2] + y[l.IDN[i][1] - vol][3];
            psidiff[0] = y[l.IUP[i][1] - vol][0] - y[l.IUP[i][1] - vol][1];
            psidiff[1] = y[l.IUP[i][1] - vol][2] - y[l.IUP[i][1] - vol][3];

            x[l.toEOflat(nt, nx)][0] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol][0] + 0.5*psidiff[0] + 0.5*psisum[0];
            x[l.toEOflat(nt, nx)][2] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol][2] + 0.5*psidiff[1] + 0.5*psisum[1];
            x[l.toEOflat(nt, nx)][1] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol][1] - 0.5*psidiff[0] + 0.5*psisum[0];
            x[l.toEOflat(nt, nx)][3] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol][3] - 0.5*psidiff[1] + 0.5*psisum[1];

        } else {

            psisum[0] = y[l.IUP[i][1] - vol][0] + y[l.IUP[i][1] - vol][1];
            psisum[1] = y[l.IUP[i][1] - vol][2] + y[l.IUP[i][1] - vol][3];
            psidiff[0] = y[l.IDN[i][1] - vol][0] - y[l.IDN[i][1] - vol][1];
            psidiff[1] = y[l.IDN[i][1] - vol][2] - y[l.IDN[i][1] - vol][3];

            x[l.toEOflat(nt, nx)][0] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol][0] + 0.5 * psisum[0] + 0.5 * psidiff[0];
            x[l.toEOflat(nt, nx)][2] -=  sgn[0] * 1.0 * y[l.IUP[i][0] - vol][2] + 0.5 * psisum[1] + 0.5 * psidiff[1];
            x[l.toEOflat(nt, nx)][1] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol][1] + 0.5 * psisum[0] - 0.5 * psidiff[0];
            x[l.toEOflat(nt, nx)][3] -=  sgn[1] * 1.0 * y[l.IDN[i][0] - vol][3] + 0.5 * psisum[1] - 0.5 * psidiff[1];

        }
                                            
    }

}
void DiracOP::D_oe(std::vector<vec_fc>::iterator y, std::vector<vec_fc>::iterator x, bool const dagger){
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Hopping term
    int const Nt=l.Nt, Nx=l.Nx;

    double sgn[2];
    int nt, nx;

    int i;
    for(int j=0; j<vol; j++){

        i = j+vol; // full index (for lookup tables)

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<double> psisum[2], psidiff[2];
        
        if (dagger) {

            psisum[0] = y[l.IDN[i][1]][0] + y[l.IDN[i][1]][1];
            psisum[1] = y[l.IDN[i][1]][2] + y[l.IDN[i][1]][3];
            psidiff[0] = y[l.IUP[i][1]][0] - y[l.IUP[i][1]][1];
            psidiff[1] = y[l.IUP[i][1]][2] - y[l.IUP[i][1]][3];

            x[j][0] -=  sgn[1] * 1.0 * y[l.IDN[i][0]][0] + 0.5*psidiff[0] + 0.5*psisum[0];
            x[j][2] -=  sgn[1] * 1.0 * y[l.IDN[i][0]][2] + 0.5*psidiff[1] + 0.5*psisum[1];
            x[j][1] -=  sgn[0] * 1.0 * y[l.IUP[i][0]][1] - 0.5*psidiff[0] + 0.5*psisum[0];
            x[j][3] -=  sgn[0] * 1.0 * y[l.IUP[i][0]][3] - 0.5*psidiff[1] + 0.5*psisum[1];

        } else {

            psisum[0] = y[l.IUP[i][1]][0] + y[l.IUP[i][1]][1];
            psisum[1] = y[l.IUP[i][1]][2] + y[l.IUP[i][1]][3];
            psidiff[0] = y[l.IDN[i][1]][0] - y[l.IDN[i][1]][1];
            psidiff[1] = y[l.IDN[i][1]][2] - y[l.IDN[i][1]][3];

            x[j][0] -=  sgn[0] * 1.0 * y[l.IUP[i][0]][0] + 0.5 * psisum[0] + 0.5 * psidiff[0];
            x[j][2] -=  sgn[0] * 1.0 * y[l.IUP[i][0]][2] + 0.5 * psisum[1] + 0.5 * psidiff[1];
            x[j][1] -=  sgn[1] * 1.0 * y[l.IDN[i][0]][1] + 0.5 * psisum[0] - 0.5 * psidiff[0];
            x[j][3] -=  sgn[1] * 1.0 * y[l.IDN[i][0]][3] + 0.5 * psisum[1] - 0.5 * psidiff[1];


        }
                                            
    }
}

void DiracOP::applyLTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res){
    std::vector<vec_fc> temp(l.vol/2), temp2(l.vol/2);
    for(int i=0; i<l.vol/2; i++) {temp[i].setZero(); temp2[i].setZero();}
    
    D_oo_inv(vec + l.vol/2, temp.begin());
    D_eo(temp.begin(), temp2.begin());
    
    for(int i=0; i<l.vol/2; i++){
        res[i] = vec[i] + temp2[i];
        res[l.vol/2+i] = vec[l.vol/2+i];
    }
}

void DiracOP::applyRTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res){
    std::vector<vec_fc> temp(l.vol/2), temp2(l.vol/2);
    for(int i=0; i<l.vol/2; i++) {temp[i].setZero(); temp2[i].setZero();}
    
    D_oe(vec, temp.begin());
    D_oo_inv(temp.begin(), temp2.begin());

    for(int i=0; i<l.vol/2; i++){
        res[i] = vec[i];
        res[l.vol/2+i] = vec[l.vol/2+i] + temp2[i];
    }
}

void DiracOP::applyLinvTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res){
    std::vector<vec_fc> temp(l.vol/2), temp2(l.vol/2);
    for(int i=0; i<l.vol/2; i++) {temp[i].setZero(); temp2[i].setZero();}
    
    D_oo_inv(vec + l.vol/2, temp.begin());
    D_eo(temp.begin(), temp2.begin());
    
    for(int i=0; i<l.vol/2; i++){
        res[i] = vec[i] - temp2[i];
        res[l.vol/2+i] = vec[l.vol/2+i];
    }
}

void DiracOP::applyRinvTo(std::vector<vec_fc>::iterator vec, std::vector<vec_fc>::iterator res){
    std::vector<vec_fc> temp(l.vol/2), temp2(l.vol/2);
    for(int i=0; i<l.vol/2; i++) {temp[i].setZero(); temp2[i].setZero();}
    
    D_oe(vec, temp.begin());
    D_oo_inv(temp.begin(), temp2.begin());
    
    for(int i=0; i<l.vol/2; i++){
        res[i] = vec[i];
        res[l.vol/2+i] = vec[l.vol/2+i] - temp2[i];
    }
}


void DiracOP::D_ee_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger){

    float sigma;
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        idx = l.eoToVec(i);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace().real();
        x[i] += (float) (2.0 + M + mesons->g*sigma) * y[i];
        if (dagger)
            x[i] += buildCompositeOP_single(mesons->g * (mesons->M[idx[0]][idx[1]].cast<std::complex<float>>() - sigma*mat_single::Identity()).adjoint(), gamma5.cast<std::complex<float>>()) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            x[i] += buildCompositeOP_single(mesons->g * (mesons->M[idx[0]][idx[1]].cast<std::complex<float>>() - sigma*mat_single::Identity()), gamma5.cast<std::complex<float>>()) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)

    }

}

void DiracOP::D_oo_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger){
    float sigma;
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Diagonal term
    for(int i=0; i<vol; i++){
        idx = l.eoToVec(i+vol);
        sigma = 0.5 * (Pauli.tau0*mesons->M[idx[0]][idx[1]]).trace().real();
        x[i] += (2.0 + M + mesons->g*sigma) * y[i];
        if (dagger)
            x[i] += buildCompositeOP_single(mesons->g * (mesons->M[idx[0]][idx[1]].cast<std::complex<float>>() - sigma*mat_single::Identity()).adjoint(), gamma5.cast<std::complex<float>>()) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)
        else
            x[i] += buildCompositeOP_single(mesons->g * (mesons->M[idx[0]][idx[1]].cast<std::complex<float>>() - sigma*mat_single::Identity()), gamma5.cast<std::complex<float>>()) * y[i]; // flavour mixing term (just remove diagonal and tensor with gamma5)

    }
}


void DiracOP::D_eo_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger){

    std::vector<int> idx(2);
    int const vol = l.vol/2;


    int const Nt=l.Nt, Nx=l.Nx;

    std::complex<float> sgn[2];
    int nt, nx;
    
    for(int i=0; i<vol; i++){

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<float> psisum[2], psidiff[2];

        // maybe use ? : syntax to remove the if (more elegant?)
        
        if (dagger) {

            psisum[0] = y[l.IDN[i][1] - vol][0] + y[l.IDN[i][1] - vol][1];
            psisum[1] = y[l.IDN[i][1] - vol][2] + y[l.IDN[i][1] - vol][3];
            psidiff[0] = y[l.IUP[i][1] - vol][0] - y[l.IUP[i][1] - vol][1];
            psidiff[1] = y[l.IUP[i][1] - vol][2] - y[l.IUP[i][1] - vol][3];

            x[l.toEOflat(nt, nx)][0] -=  sgn[1] * (std::complex<float>) 1.0 * y[l.IDN[i][0] - vol][0] + (std::complex<float>) 0.5*psidiff[0] + (std::complex<float>) 0.5*psisum[0];
            x[l.toEOflat(nt, nx)][2] -=  sgn[1] * (std::complex<float>) 1.0 * y[l.IDN[i][0] - vol][2] + (std::complex<float>) 0.5*psidiff[1] + (std::complex<float>) 0.5*psisum[1];
            x[l.toEOflat(nt, nx)][1] -=  sgn[0] * (std::complex<float>) 1.0 * y[l.IUP[i][0] - vol][1] - (std::complex<float>) 0.5*psidiff[0] + (std::complex<float>) 0.5*psisum[0];
            x[l.toEOflat(nt, nx)][3] -=  sgn[0] * (std::complex<float>) 1.0 * y[l.IUP[i][0] - vol][3] - (std::complex<float>) 0.5*psidiff[1] + (std::complex<float>) 0.5*psisum[1];

        } else {

            psisum[0] = y[l.IUP[i][1] - vol][0] + y[l.IUP[i][1] - vol][1];
            psisum[1] = y[l.IUP[i][1] - vol][2] + y[l.IUP[i][1] - vol][3];
            psidiff[0] = y[l.IDN[i][1] - vol][0] - y[l.IDN[i][1] - vol][1];
            psidiff[1] = y[l.IDN[i][1] - vol][2] - y[l.IDN[i][1] - vol][3];

            x[l.toEOflat(nt, nx)][0] -=  sgn[0] * (std::complex<float>) 1.0 * y[l.IUP[i][0] - vol][0] + (std::complex<float>) 0.5 * psisum[0] + (std::complex<float>) 0.5 * psidiff[0];
            x[l.toEOflat(nt, nx)][2] -=  sgn[0] * (std::complex<float>) 1.0 * y[l.IUP[i][0] - vol][2] + (std::complex<float>) 0.5 * psisum[1] + (std::complex<float>) 0.5 * psidiff[1];
            x[l.toEOflat(nt, nx)][1] -=  sgn[1] * (std::complex<float>) 1.0 * y[l.IDN[i][0] - vol][1] + (std::complex<float>) 0.5 * psisum[0] - (std::complex<float>) 0.5 * psidiff[0];
            x[l.toEOflat(nt, nx)][3] -=  sgn[1] * (std::complex<float>) 1.0 * y[l.IDN[i][0] - vol][3] + (std::complex<float>) 0.5 * psisum[1] - (std::complex<float>) 0.5 * psidiff[1];

        }
                                            
    }

}
void DiracOP::D_oe_single(std::vector<vec_fc_single>::iterator y, std::vector<vec_fc_single>::iterator x, bool const dagger){
    std::vector<int> idx(2);
    int const vol = l.vol/2;

    // Hopping term
    int const Nt=l.Nt, Nx=l.Nx;

    float sgn[2];
    int nt, nx;

    int i;
    for(int j=0; j<vol; j++){

        i = j+vol; // full index (for lookup tables)

        idx = l.eoToVec(i);
        nt = idx[0]; nx = idx[1];
        sgn[0] = (nt == (Nt-1)) ? -1.0 : 1.0;
        sgn[1] = (nt == 0) ? -1.0 : 1.0;

        std::complex<float> psisum[2], psidiff[2];
        
        if (dagger) {

            psisum[0] = y[l.IDN[i][1]][0] + y[l.IDN[i][1]][1];
            psisum[1] = y[l.IDN[i][1]][2] + y[l.IDN[i][1]][3];
            psidiff[0] = y[l.IUP[i][1]][0] - y[l.IUP[i][1]][1];
            psidiff[1] = y[l.IUP[i][1]][2] - y[l.IUP[i][1]][3];

            x[j][0] -=  sgn[1] * (std::complex<float>) 1.0 * y[l.IDN[i][0]][0] + (std::complex<float>) 0.5*psidiff[0] + (std::complex<float>) 0.5*psisum[0];
            x[j][2] -=  sgn[1] * (std::complex<float>) 1.0 * y[l.IDN[i][0]][2] + (std::complex<float>) 0.5*psidiff[1] + (std::complex<float>) 0.5*psisum[1];
            x[j][1] -=  sgn[0] * (std::complex<float>) 1.0 * y[l.IUP[i][0]][1] - (std::complex<float>) 0.5*psidiff[0] + (std::complex<float>) 0.5*psisum[0];
            x[j][3] -=  sgn[0] * (std::complex<float>) 1.0 * y[l.IUP[i][0]][3] - (std::complex<float>) 0.5*psidiff[1] + (std::complex<float>) 0.5*psisum[1];

        } else {

            psisum[0] = y[l.IUP[i][1]][0] + y[l.IUP[i][1]][1];
            psisum[1] = y[l.IUP[i][1]][2] + y[l.IUP[i][1]][3];
            psidiff[0] = y[l.IDN[i][1]][0] - y[l.IDN[i][1]][1];
            psidiff[1] = y[l.IDN[i][1]][2] - y[l.IDN[i][1]][3];

            x[j][0] -=  sgn[0] * (std::complex<float>) 1.0 * y[l.IUP[i][0]][0] + (std::complex<float>) 0.5 * psisum[0] + (std::complex<float>) 0.5 * psidiff[0];
            x[j][2] -=  sgn[0] * (std::complex<float>) 1.0 * y[l.IUP[i][0]][2] + (std::complex<float>) 0.5 * psisum[1] + (std::complex<float>) 0.5 * psidiff[1];
            x[j][1] -=  sgn[1] * (std::complex<float>) 1.0 * y[l.IDN[i][0]][1] + (std::complex<float>) 0.5 * psisum[0] - (std::complex<float>) 0.5 * psidiff[0];
            x[j][3] -=  sgn[1] * (std::complex<float>) 1.0 * y[l.IDN[i][0]][3] + (std::complex<float>) 0.5 * psisum[1] - (std::complex<float>) 0.5 * psidiff[1];


        }
                                            
    }
}

