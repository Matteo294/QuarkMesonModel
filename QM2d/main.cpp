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
//#include "functions/functions.h"
#include "ConjugateGradientSolver/ConjugateGradientSolver.h"
#include "params.h"
#include <chrono>


/// !!! check references when cycling thourgh vectors
// hardcode Nf=2
// set public-private things
// change Dee Doo ... so that result is assigned and not modified
// flatten mesons matrix and remove lattice
// faster calculation of Mbar and determinant

// remove EOdoubleCG and use only doubleCG

using namespace std;


int main(){

    Lattice lattice(Nt, Nx);
    O4Mesons mesons(meson_M2, lam, g, sigma, pi, lattice);
    SpinorField psiField(lattice.Nt*lattice.Nx);
    DiracOP Dirac(fermion_M, mesons, lattice);
    ConjugateGradientSolver CG(IterMax, tol, Dirac);

    psiField.val[0] = vec_fc {1.0, 1.0, 1.0, 1.0}; // set source for CG

    //ptrMat f = &(DiracOP::applyDhatTo);

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
   


    cout << "\nNote: remember to recompile main.cpp after changing params.h! \n\nStarting CG in mode ";

    vector<vec_fc> temp1(lattice.vol, vec_fc::Zero()), temp2(lattice.vol/2, vec_fc::Zero()), temp3(lattice.vol, vec_fc::Zero()); // useful variables
    std::vector<vec_fc_single> buf(lattice.vol, vec_fc_single::Zero()), temp_single(lattice.vol, vec_fc_single::Zero());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    switch (CGmode){
    case 0:
        cout << "double precision, no preconditioning... \n";
        CG.doubleCG_D(psiField.val.begin(), psiField.val.end(), temp1.begin());   
        Dirac.applyTo(temp1.begin(), psiField.val.begin(), MatrixType::Dagger);
        break;
    
    case 1:
        cout << "double precision, EO preconditioning... \n";

        Dirac.D_oo_inv(psiField.val.begin() + lattice.vol/2, temp2.begin());
        Dirac.D_eo(temp2.begin(), temp1.begin());

        for(int i=0; i<lattice.vol/2; i++) temp2[i] = psiField.val[i] - temp1[i];
        
        CG.doubleCG_Dhat(temp2.begin(), temp2.begin() + lattice.vol/2, temp3.begin());
        for(int i=0; i<lattice.vol/2; i++){psiField.val[i].setZero();}
        Dirac.applyDhatTo(temp3.begin(), psiField.val.begin(), MatrixType::Dagger);

        for(int i=0; i<lattice.vol; i++){temp1[i].setZero();}
        Dirac.D_oe(psiField.val.begin(), temp1.begin());

        for(int i=0; i<lattice.vol/2; i++) temp3[i] = psiField.val[lattice.vol/2 + i] - temp1[i];

        for(int i=lattice.vol/2; i<lattice.vol; i++){psiField.val[i].setZero();}
        Dirac.D_oo_inv(temp3.begin(), psiField.val.begin() + lattice.vol/2);

        break;

    case 2:
        /*cout << "single precision, no preconditioning... \n";
        for(int i=0; i<lattice.vol; i++) buf[i] = psiField.val[i].cast<std::complex<float>>();
        mesons.writeDoubleToSingle();
        CG.singleCG(buf.begin(), buf.end(), temp_single.begin());  
        for(int i=0; i<lattice.vol; i++) buf[i].setZero();
        Dirac.applyTo_single(temp_single.begin(), buf.begin(), MatrixType::Dagger);
        for(int i=0; i<lattice.vol; i++) psiField.val[i] = buf[i].cast<std::complex<double>>();*/
        break;

    /*case 4:
    // check set to zero
        cout << "mixed precision, no preconditioning... \n";
        for(int i=0; i<lattice.vol; i++) buf[i] = psiField.val[i].cast<std::complex<float>>();
        mesons.writeDoubleToSingle();
        CG.mixedCG(psiField.val.begin(), psiField.val.end(), temp3.begin(), IterMaxSingle, 1e-3);  
        for(int i=0; i<lattice.vol; i++) buf[i].setZero();
        Dirac.applyTo(temp3.begin(), temp1.begin(), 1);
        for(int i=0; i<lattice.vol; i++) psiField.val[i] = buf[i].cast<std::complex<double>>();
        break;*/

  
    default:
        cout << "CGmode not valid. Please select value 0, 1 or 2 (3, 4 and 5 not yet implemented)" << endl;
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
            corr += std::accumulate(psiField.val[lattice.toEOflat(nt, nx)].begin(), psiField.val[lattice.toEOflat(nt, nx)].end(), 0.0+0.0*im);
        }
        datafile << corr.real() << endl;
    }

    return 0;
}














