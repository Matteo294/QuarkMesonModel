#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <fstream>
#include "SpinorField/SpinorField.h"
#include "DiracOP/DiracOP.h"
#include "Mesons/O4Mesons.h"
#include "Lattice/Lattice.h"
#include "functions/functions.h"
#include "ConjugateGradientSolver/ConjugateGradientSolver.h"
#include "params.h"
#include <chrono>


// set public-private things
// change Dee Doo ... so that result is assigned and not modified
// move 2 + m into mesons

// difference between typedef and using x = y ?
// Spinor class useful?
// creating class in each file necessary?

// CHECK i factors in the calculation of N !!!!


using namespace std;


int main(){

    Lattice lattice(Nt, Nx);
    O4Mesons mesons(meson_M2, lam, g, sigma, pi, lattice);
    SpinorField psiField(lattice.Nt*lattice.Nx);
    DiracOP<vecfield_iter, double> Dirac_d(fermion_M, mesons, lattice);
    ConjugateGradientSolver<vecfield_iter, double> CGdouble(IterMax, tol, Dirac_d);

    psiField.pos[0].val = std::vector<std::complex<double>> {1.0, 1.0, 1.0, 1.0};

    // Write configuration to file
    ofstream conffile;
    conffile.open("conffile.txt");
    // Save parameters to file
    // !!! substitute these params with those stored in the objects to be sure that we are using the correct ones
    conffile    << "m0=" << fermion_M << "\n"
                << "lam=" << lam << "\n"
                << "g=" << g << "\n"
                << "sigma=" << sigma.real() << "\n"
                << "pi1=" << pi[0].real() << "\n"
                << "pi2=" << pi[1].real() << "\n"
                << "pi3=" << pi[2].real();
   
    cout << "\nNote: remember to run make clean after changing header files!! \n\nStarting CG in mode ";

    vecfield temp1(lattice.vol, Spinor_d()), temp2(lattice.vol/2, Spinor_d()), temp3(lattice.vol, Spinor_d()); // useful variables
    //vecfield buf(lattice.vol, vec_fc_single::Zero()), temp_single(lattice.vol, vec_fc_single::Zero());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    switch (CGmode){
    case 0:
        cout << "double precision, no preconditioning... \n";
        CGdouble.doubleCG_D(psiField.pos.begin(), psiField.pos.end(), temp1.begin());   
        Dirac_d.applyTo(temp1.begin(), psiField.pos.begin(), MatrixType::Dagger);
        break;
    case 1:
        cout << "double precision, EO preconditioning... \n";

        Dirac_d.D_oo_inv(psiField.pos.begin() + lattice.vol/2, temp2.begin());
        Dirac_d.D_eo(temp2.begin(), temp1.begin());

        spinorDiff(psiField.pos.begin(), psiField.pos.begin() + lattice.vol/2, temp1.begin(), temp2.begin());
        
        CGdouble.doubleCG_Dhat(temp2.begin(), temp2.end(), temp3.begin());
        for(int i=0; i<lattice.vol/2; i++) std::fill(psiField.pos[i].val.begin(), psiField.pos[i].val.begin(), 0.0);
        Dirac_d.applyDhatTo(temp3.begin(), psiField.pos.begin(), MatrixType::Dagger);

        std::fill(temp1.begin(), temp1.end(), Spinor_d());
        Dirac_d.D_oe(psiField.pos.begin(), temp1.begin());

        spinorDiff(psiField.pos.begin() + lattice.vol/2, psiField.pos.end(), temp1.begin(), temp3.begin());

        for(int i=lattice.vol/2; i<lattice.vol; i++) std::fill(psiField.pos[i].val.begin(), psiField.pos[i].val.end(), 0.0);
        Dirac_d.D_oo_inv(temp3.begin(), psiField.pos.begin() + lattice.vol/2);

        break;
    }   

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    // Correlator to extract masses
    ofstream datafile;
    datafile.open("data.csv");
    datafile << "f0c0,f0c1,f1c0,f1c1" << endl;
    complex<double> corr = 0.0;
    for(int nt=0; nt<Nt; nt++){
        corr = 0.0;
        for(int nx=0; nx<Nx; nx++){
            corr += std::accumulate(psiField.pos[lattice.toEOflat(nt, nx)].val.begin(), psiField.pos[lattice.toEOflat(nt, nx)].val.end(), 0.0+0.0*im);
        }
        datafile << corr.real() << endl;
    }

    return 0;
}














