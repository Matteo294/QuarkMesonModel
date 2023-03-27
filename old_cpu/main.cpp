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

// !!!!!!!!!! IUP AND IDN INVERTED !!!!!!!!!!!!!!

// set public-private things
// move 2 + m into mesons

// difference between typedef and using x = y ?
// Spinor class useful?

using namespace std;


int main(int argc, char *argv[]){

    if (argc == 2) CGmode = stoi(argv[1]);

    Lattice lattice(Nt, Nx);
    SpinorField psiField(lattice.Nt*lattice.Nx);

    // Double precision
    O4Mesons<double> mesons_d(meson_M2, lam, g, sigma, pi, lattice);
    DiracOP<vecfield_iter, double> Dirac_d(fermion_M, mesons_d, lattice);
    ConjugateGradientSolver<vecfield_iter, double> CGdouble(IterMax, tol, Dirac_d);
    // Single precision
    O4Mesons<float> mesons_f(meson_M2, lam, g, sigma, pi, lattice);
    DiracOP<vecfield_single_iter, float> Dirac_f(fermion_M, mesons_f, lattice);
    ConjugateGradientSolver<vecfield_single_iter, float> CGsingle(IterMax, tol, Dirac_f);

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
    vecfield_single buffer1(lattice.vol, Spinor<float>()), buffer2(lattice.vol, Spinor<float>());

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    switch (CGmode){
    case 0:
        cout << "double precision, no preconditioning... \n";
        CGdouble.solve_D(psiField.pos.begin(), psiField.pos.end(), temp1.begin());   
        Dirac_d.applyTo(temp1.begin(), psiField.pos.begin(), MatrixType::Dagger);
        break;

    case 1:
        cout << "double precision, EO preconditioning... \n";

        Dirac_d.D_oo_inv(psiField.pos.begin() + lattice.vol/2, temp2.begin());
        Dirac_d.D_eo(temp2.begin(), temp1.begin());

        spinorDiff(psiField.pos.begin(), psiField.pos.begin() + lattice.vol/2, temp1.begin(), temp2.begin());
        
        CGdouble.solve_Dhat(temp2.begin(), temp2.end(), temp3.begin());
        for(int i=0; i<lattice.vol/2; i++) std::fill(psiField.pos[i].val.begin(), psiField.pos[i].val.begin(), 0.0);
        Dirac_d.applyDhatTo(temp3.begin(), psiField.pos.begin(), MatrixType::Dagger);

        std::fill(temp1.begin(), temp1.end(), Spinor_d());
        Dirac_d.D_oe(psiField.pos.begin(), temp1.begin());

        spinorDiff(psiField.pos.begin() + lattice.vol/2, psiField.pos.end(), temp1.begin(), temp3.begin());

        for(int i=lattice.vol/2; i<lattice.vol; i++) std::fill(psiField.pos[i].val.begin(), psiField.pos[i].val.end(), 0.0);
        Dirac_d.D_oo_inv(temp3.begin(), psiField.pos.begin() + lattice.vol/2);

        break;

    case 2:
        cout << "single precision, no preconditioning... \n";
        for (int i=0; i<lattice.vol; i++){
            std::transform(psiField.pos[i].val.begin(), psiField.pos[i].val.end(), buffer1[i].val.begin(), [](std::complex<double> x) { return (std::complex<float>)x;});
            std::transform(temp1[i].val.begin(), temp1[i].val.end(), buffer2[i].val.begin(), [](std::complex<double> x) { return (std::complex<float>)x;});
        }
        CGsingle.solve_D(buffer1.begin(), buffer1.end(), buffer2.begin());   
        Dirac_f.applyTo(buffer2.begin(), buffer1.begin(), MatrixType::Dagger);
        for (int i=0; i<lattice.vol; i++){
            std::transform(buffer1[i].val.begin(), buffer1[i].val.end(), psiField.pos[i].val.begin(), [](std::complex<float> x) { return (std::complex<double>)x;});
        }
        break;
    default:
    cout << "\n CG mode not valid \n";
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














