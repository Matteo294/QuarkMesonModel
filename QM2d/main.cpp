#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <complex>
#include <random>
#include "stuff.h"



/*** In this code the chiral (or Weyl) basis is used ***/
// for free fermions the Dirac operator is diagonal in flavour space so we can store them in two separate arrays

using namespace std;
using mat2 = Eigen::Matrix2cd;
using vec2 = Eigen::Vector2cd; 
using fieldMat = Eigen::Matrix<vec2, Nt, Nx>;

complex<double> const im {0, 1};
double const m = 1.0; // fermions mass
int const Nmax = 1000; // Maximum number of CG iterations
int const Nf=2; // Number of flavours
double const eps=1e-3; // CG precision

// gamma matrices Minkowski metric
mat2 const gamma0 {{1, 0}, {0, -1}};
mat2 const gamma1 {{0, 1}, {-1, 0}};
mat2 const Id2 {{1.0, 0.0}, {0.0, 1.0}};

double const p=2.0*3.0*M_PI/Nt, q=2.0*6.0*M_PI/Nx; // useful to test Dirac operator

typedef class SpinorField {
    public: 
        int const Nt, Nx;
        SpinorField(int const Nt, int const Nx); // Construct by specifying dimension
        SpinorField(fieldMat& psi); // Construct by specifying fields value (needed for the overload of "-" operator)
        ~SpinorField() {;}
        fieldMat psi;
    private:
        random_device rnddev;
        mt19937 rndgen;
        normal_distribution<double> dist;
} SpinorField;

SpinorField* DiracmatSpinorProduct(SpinorField const& s); // applies dirac operator to s
int PBC(int n, int const N){return (n+N)%N;}


int main(){

    /************************************************************************/
    cout << "Checking Clifford algebra in Euclidean metric: \n\n" ;
    cout << "{gamma0,gamma1} = \n" << gamma0*gamma1 + gamma1*gamma0 << "\n\n";
    cout << "{gamma1,gamma0} = \n"<< gamma1*gamma0 + gamma0*gamma1 << "\n\n";
    cout << "{gamma0,gamma0} = \n"<< gamma0*gamma0 + gamma0*gamma0 << "\n\n";
    cout << "{gamma1,gamma1} = \n"<< gamma1*gamma1 + gamma1*gamma1 << "\n\n";
    //mat2 gamma5 = 0.5*im*(gamma0*gamma1-gamma1*gamma0);
    mat2 gamma5 = gamma0*gamma1;
    cout << gamma5 << endl;
    cout << "PL = (1-gamma5)/2 = \n" << 0.5 * (Id2 - gamma5) << "\n\n";
    cout << "PR = (1+gamma5)/2 = \n" << 0.5 * (Id2 + gamma5) << "\n\n";
    /************************************************************************/

    SpinorField psiField(Nt, Nx); 
    SpinorField* psiFieldNew = DiracmatSpinorProduct(psiField);
    
    // Test Dirac operator (see notes for details on what is going on here)
    complex<double> dcp[2];
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            dcp[0] = (2.0-cos(p)-cos(q)+im*sin(p))*psiField.psi(nt,nx)(0) + im*sin(q)*psiField.psi(nt,nx)(1);
            dcp[1] = (2.0-cos(p)-cos(q)-im*sin(p))*psiField.psi(nt,nx)(1) - im*sin(q)*psiField.psi(nt,nx)(0);
            cout << psiFieldNew->psi(nt,nx)(0) << " " << dcp[0] << endl;
        }
    }

    return 0;
}

SpinorField::SpinorField(int const Nt, int const Nx) :
    rndgen(rnddev()),
    dist(0., 1.),
    Nt{Nt},
    Nx{Nx}
{
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            psi(nt,nx) = vec2 {1.0, 1.3} * exp(im*p*(double)nt+im*q*(double)nx);
        }
    }
}

SpinorField::SpinorField(fieldMat& p) :
    rndgen(rnddev()),
    dist(0., 1.),
    Nt{p.cols()},
    Nx{p.rows()}
{
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            psi(nt,nx) = p(nt,nx);
        }
    }
}

SpinorField* DiracmatSpinorProduct(SpinorField const & s){
    fieldMat r;
    double sgn[2];
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            sgn[0] = (nt+1 >= Nt) || (nt < 0) ? -1.0 : 1.0;
            sgn[1] = (nt-1 >= Nt) || (nt < 0) ? -1.0 : 1.0;
            r(nt, nx)(0) = 2.0*s.psi(nt, nx)(0) - s.psi(PBC(nt-1, Nt), nx)(0) - 0.5*(s.psi(nt, PBC(nx-1, Nx))(0) + s.psi(nt, PBC(nx-1, Nx))(1)) - 0.5*(s.psi(nt, PBC(nx+1, Nx))(0) - s.psi(nt, PBC(nx+1, Nx))(1));
            r(nt, nx)(1) = 2.0*s.psi(nt, nx)(1) - s.psi(PBC(nt+1, Nt), nx)(1) - 0.5*(s.psi(nt, PBC(nx-1, Nx))(1) - s.psi(nt, PBC(nx-1, Nx))(0)) - 0.5*(s.psi(nt, PBC(nx+1, Nx))(0) + s.psi(nt, PBC(nx+1, Nx))(1));
        }
    }
    return new SpinorField(r);
}








