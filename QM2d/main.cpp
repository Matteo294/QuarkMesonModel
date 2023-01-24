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

using namespace std;

int main(){ 

    // Create mesons, Langevin simulator, fermion field and a Dirac operator
    O4Mesons mesons(Nt, Nx, meson_M2, lam, g, sigma, pi);
    Langevin langevin(&mesons); // for the moment only for mesonic sector (no yukawa)
    SpinorField psiField(Nt, Nx, Nf);
    DiracOP Dirac(fermion_M, &mesons);


    // Write configuration to file
    ofstream conffile;
    conffile.open("conffile.txt");

    // Save parameters to file
    // !!! substitute these params with those stored in the objects to be sure that we are using the correct ones
    conffile    << "m0=" << fermion_M << "\n"
                << "lam=" << lam << "\n"
                << "g=" << g << "\n"
                << "sigma=" << sigma << "\n"
                << "pi1=" << pi[0] << "\n"
                << "pi2=" << pi[1] << "\n"
                << "pi3=" << pi[2];

    // Perform CG to get the correlator from which we then extract the mass
    SpinorField afterCG(Nt, Nx, Nf);
    CG(psiField, afterCG, Dirac);   
    psiField = Dirac.applyToCSR(afterCG, 1);

    // Thermalisation
    /*for(int n=0; n<Ntherm; n++){
        langevin.LangevinRun(0.01, 1.0);
    }
    cout << "Thermalisation done" << endl;

    // Calculate stuff
    double M = 0.0;
    for(int n=0; n<Ndata; n++){
        langevin.LangevinRun(0.01, 1.0);
        M += (double) mesons.norm() / (Nt*Nx);
    }
    cout << "Magnetization: " << M/3000.0;
    if (meson_M2 < 0) cout << "\t expected: " << sqrt(-6*meson_M2/lam) << endl;*/

    // Correlator to extract masses
    ofstream datafile;
    datafile.open("data.csv");
    datafile << "f0c0,f0c1,f1c0,f1c1" << endl;
    complex<double> corr = 0.0;
    for(int nt=0; nt<Nt; nt++){
        corr = 0.0;
        for(int nx=0; nx<Nx; nx++){
            corr += std::accumulate(psiField.val[toEOflat(nt, nx)].begin(), psiField.val[toEOflat(nt, nx)].end(), 0.0+0.0*im);
        }
        datafile << corr.real() << endl;
    }

    return 0;
}














