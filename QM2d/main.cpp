#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <fstream>
#include <algorithm>
#include "SpinorField/SpinorField.h"
#include "DiracOP/DiracOP.h"
#include "Mesons/O4Mesons.h"
#include "Lattice/Lattice.h"
#include "ConjugateGradientSolver/ConjugateGradientSolver.h"
#include "params.h"
#include <chrono>


/// !!! check references when cycling thourgh vectors
// hardcode Nf=2
// set public-private things

using namespace std;

int main(){ 

    // Create lattice, mesons, fermions and a Dirac operator
    Lattice lattice(Nt, Nx);
    O4Mesons mesons(meson_M2, lam, g, sigma, pi, lattice);
    SpinorField psiField(lattice);
    DiracOP Dirac(fermion_M, &mesons, lattice);
    ConjugateGradientSolver CG(IterMax, tol, Dirac);

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
    /*SpinorField afterCG(lattice);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    CG.baseCG(psiField, afterCG);   
    psiField = Dirac.applyTo(afterCG, 1);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    // Correlator to extract masses
    ofstream datafile;
    datafile.open("data.csv");
    datafile << "f0c0,f0c1,f1c0,f1c1" << endl;
    complex<double> corr = 0.0;
    for(int nt=0; nt<Nt; nt++){
        corr = 0.0;
        for(int nx=0; nx<Nx; nx++){
            corr += std::accumulate(psiField.val[lattice.toEOflat(nt, nx)].begin(), psiField.val[lattice.toEOflat(nt, nx)].end(), 0.0+0.0*im);
        }
        datafile << corr.real() << endl;
    }*/

    // Perform EO preconditioned CG
    SpinorField afterCG(lattice);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    psiField = Dirac.applyLto(psiField, 0);
    CG.eoCG(psiField, afterCG); 
    psiField = Dirac.applyLDRTo(afterCG, 1);
    psiField = Dirac.applyRto(psiField, 0);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    // Correlator to extract masses
    ofstream datafile;
    datafile.open("data.csv");
    datafile << "f0c0,f0c1,f1c0,f1c1" << endl;
    complex<double> corr = 0.0;
    for(int nt=0; nt<Nt; nt++){
        corr = 0.0;
        for(int nx=0; nx<Nx; nx++){
            corr += std::accumulate(psiField.val[lattice.toEOflat(nt, nx)].begin(), psiField.val[lattice.toEOflat(nt, nx)].end(), 0.0+0.0*im);
        }
        datafile << corr.real() << endl;
    }

    return 0;
}














