#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <fstream>
#include <algorithm>
#include "SpinorField/SpinorField.h"
#include "DiracOP/DiracOP.h"
#include "Mesons/O4Mesons.h"
#include "Langevin/Langevin.h"


/// !!! check references when cycling thourgh vectors
/// !!! pass mass to constructor and store it in the class

using namespace std;

void CG(SpinorField const& inPsi, SpinorField& outV);


int main(){ 

    // check EO indexing
    /*vector<int> idx(4);
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            for(int f=0; f<Nf; f++){
                for(int c=0; c<2; c++){
                    cout << nt << " " << nx << " " << f << " " << c << "\n";
                    idx = eoToVec(toEOflat(nt, nx, f, c));
                    cout << idx[0] << " " << idx[1] << " " << idx[2] << " " << idx[3] << "\n \n";
                }
            }
        }
    }*/

    O4Mesons mesons(Nt, Nx, meson_M2, lam);
    Langevin langevin(&mesons);
    SpinorField psiField(Nt, Nx, Nf), afterCG(Nt, Nx, Nf);
    DiracOP Dirac(fermion_M, &mesons);

    CG(psiField, afterCG, Dirac);   
    psiField = Dirac.applyTo(afterCG, 1);


    ofstream datafile;
    datafile.open("data.csv");
    datafile << "psi1,psi2" << endl;

    /*for(int n=0; n<1000; n++){
        langevin.LangevinRun(0.01, 1.0);
    }*/

    vector<int> idx (4); // Nt, Nx, Nf, c
    vector<vector<double>> correlator_psi1(Nt, vector<double> (Nx, 0.0)); // first component
    auto correlator_psi2 = correlator_psi1; // second component
    for(int i=0; i<vol; i++){
        idx = eoToVec(i);
        if ((idx[3] == 0) && (idx[2] == 0)) correlator_psi1[idx[0]][idx[1]] = psiField.val[i].real() + psiField.val[i+1].real();
    }
    
    for(int nt=0; nt<Nt; nt++) 
        datafile    << accumulate(correlator_psi1[nt].begin(), correlator_psi1[nt].end(), 0.0) << "," 
                    << accumulate(correlator_psi2[nt].begin(), correlator_psi2[nt].end(), 0.0) << "\n";
    

    return 0;
}














