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


/// !!! check references when cycling through vectors
// set public-private things
// change Dee Doo ... so that result is assigned and not modified

// !!!!!!!!!!!!!!!!!!!!!!!!!! start checking if Doo*Dooinv = 1


// remove EOdoubleCG and use only doubleCG

using namespace std;


int main(){

    Lattice lattice(Nt, Nx);
    O4Mesons mesons(meson_M2, lam, g, sigma, pi, lattice);
    SpinorField psiField(lattice.Nt*lattice.Nx);
    DiracOP Dirac(fermion_M, mesons, lattice);
    ConjugateGradientSolver CG(IterMax, tol, Dirac);

    psiField.psi[0].val = std::vector<std::complex<double>> {1.0, 1.0, 1.0, 1.0};

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
   
    cout << "\nNote: remember to recompile main.cpp after changing params.h! \n\nStarting CG in mode ";

    vecfield temp1(lattice.vol, Spinor()), temp2(lattice.vol/2, Spinor()), temp3(lattice.vol, Spinor()); // useful variables
    //vecfield buf(lattice.vol, vec_fc_single::Zero()), temp_single(lattice.vol, vec_fc_single::Zero());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    switch (CGmode){
    case 0:
        cout << "double precision, no preconditioning... \n";
        CG.doubleCG_D(psiField.psi.begin(), psiField.psi.end(), temp1.begin());   
        Dirac.applyTo(temp1.begin(), psiField.psi.begin(), MatrixType::Dagger);
        break;
    case 1:
        cout << "double precision, EO preconditioning... \n";

        Dirac.D_oo_inv(psiField.psi.begin() + lattice.vol/2, temp2.begin());
        Dirac.D_eo(temp2.begin(), temp1.begin());

        spinorDiff(psiField.psi.begin(), psiField.psi.begin() + lattice.vol/2, temp1.begin(), temp2.begin());
        
        CG.doubleCG_Dhat(temp2.begin(), temp2.end(), temp3.begin());
        for(int i=0; i<lattice.vol/2; i++) std::fill(psiField.psi[i].val.begin(), psiField.psi[i].val.begin(), 0.0);
        Dirac.applyDhatTo(temp3.begin(), psiField.psi.begin(), MatrixType::Dagger);

        std::fill(temp1.begin(), temp1.end(), Spinor());
        Dirac.D_oe(psiField.psi.begin(), temp1.begin());

        spinorDiff(psiField.psi.begin() + lattice.vol/2, psiField.psi.end(), temp1.begin(), temp3.begin());

        for(int i=lattice.vol/2; i<lattice.vol; i++) std::fill(psiField.psi[i].val.begin(), psiField.psi[i].val.end(), 0.0);
        Dirac.D_oo_inv(temp3.begin(), psiField.psi.begin() + lattice.vol/2);

        break;
    }
    
    /*case 2:
        cout << "single precision, no preconditioning... \n";
        for(int i=0; i<lattice.vol; i++) buf[i] = psiField.val[i].cast<std::complex<float>>();
        mesons.writeDoubleToSingle();
        CG.singleCG(buf.begin(), buf.end(), temp_single.begin());  
        for(int i=0; i<lattice.vol; i++) buf[i].setZero();
        Dirac.applyTo_single(temp_single.begin(), buf.begin(), MatrixType::Dagger);
        for(int i=0; i<lattice.vol; i++) psiField.val[i] = buf[i].cast<std::complex<double>>();
        break;

    case 4:
    // check set to zero
        cout << "mixed precision, no preconditioning... \n";
        for(int i=0; i<lattice.vol; i++) buf[i] = psiField.val[i].cast<std::complex<float>>();
        mesons.writeDoubleToSingle();
        CG.mixedCG(psiField.val.begin(), psiField.val.end(), temp3.begin(), IterMaxSingle, 1e-3);  
        for(int i=0; i<lattice.vol; i++) buf[i].setZero();
        Dirac.applyTo(temp3.begin(), temp1.begin(), 1);
        for(int i=0; i<lattice.vol; i++) psiField.val[i] = buf[i].cast<std::complex<double>>();
        break;

  
    default:
        cout << "CGmode not valid. Please select value 0, 1 or 2 (3, 4 and 5 not yet implemented)" << endl;
        break;
    }
    */

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
            corr += std::accumulate(psiField.psi[lattice.toEOflat(nt, nx)].val.begin(), psiField.psi[lattice.toEOflat(nt, nx)].val.end(), 0.0+0.0*im);
        }
        datafile << corr.real() << endl;
    }

    return 0;
}














