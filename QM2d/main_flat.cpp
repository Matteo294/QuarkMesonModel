#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <fstream>

#include <algorithm>
#include "SpinorField/SpinorField.h"
#include "stuff.h"


/// !!! check references when cycling thourgh vectors
/// !!! pass mass to constructor and store it in the class

using namespace std;

void CG(SpinorField const& inPsi, SpinorField& outV);


int main(){ 

    SpinorField psiField(Nt, Nx, Nf), psiField2(Nt, Nx, Nf), afterCG(Nt, Nx, Nf);

    CG(psiField, afterCG);
    afterCG.DiracSpinorProduct(psiField2, 1);
    vector<int> idx (4); // Nt, Nx, Nf, c
    vector<vector<double>> corr(Nt, vector<double> (Nx, 0.0));
    for(int i=0; i<psiField2.volume; i++){
        idx = toVec(i);
        if (idx[3] == 0 && idx[2] == 0) corr[idx[0]][idx[1]] = psiField2.val[i].real();
    }
    ofstream outfile;
    outfile.open("data.csv");
    outfile << "data" << endl;
    for (auto& el : corr) outfile << accumulate(el.begin(), el.end(), 0.0)  << endl;

    for(int i=0; i<psiField2.volume; i++){
        idx = toVec(i);
        if (idx[3] == 1 && idx[2] == 0) corr[idx[0]][idx[1]] = psiField2.val[i].real();
    }
    for (auto& el : corr) outfile << accumulate(el.begin(), el.end(), 0.0)  << endl;


    return 0;
}




void CG(SpinorField const& inPsi, SpinorField& outPsi){
    assert(inPsi.Nt == outPsi.Nt && inPsi.Nx == outPsi.Nx && inPsi.Nf == outPsi.Nf);
    
    SpinorField 
        r(inPsi.Nt, inPsi.Nx, inPsi.Nf), // residual
        p(inPsi.Nt, inPsi.Nx, inPsi.Nf), 
        _temp(inPsi.Nt, inPsi.Nx, inPsi.Nf), // Store result of Ddagger psi
        temp(inPsi.Nt, inPsi.Nx, inPsi.Nf); // Store results of D Ddagger psi = D _temp
    complex<double> alpha;
    double beta, rmodsq;
    
    for(auto& v: outPsi.val) v = 0.0;
    r.val = inPsi.val;
    p.val = r.val;
    rmodsq = (r.dot(r)).real();

    double err;

    for(int k=0; k<IterMax; k++){

        cout << "iter " << k << ": \n";

        p.DiracSpinorProduct(_temp, 1); // apply Ddagger
        _temp.DiracSpinorProduct(temp); // apply D

        alpha = rmodsq / (p.dot(temp)).real(); 

        for(int i=0; i<inPsi.volume; i++){
            outPsi.val[i] += alpha * p.val[i];
            r.val[i] -= alpha * temp.val[i]; 
        }
        err = (r.dot(r)).real();
        cout << "Error: " << sqrt(err) << endl;
        if (sqrt(err) < tol) break;
        else{
            beta = (r.dot(r)).real() / rmodsq;
            for(int i=0; i<inPsi.volume; i++) p.val[i] = r.val[i] + beta*p.val[i];
            rmodsq = (r.dot(r)).real();
        }
    }
    cout << endl;
}









