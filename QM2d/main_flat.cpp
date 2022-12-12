#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <complex>
#include <random>
#include <fstream>


/// !!!!!!!!!!!!!! Remember to add APBE


int const Nt = 4, Nx = 6;

using namespace std;
using mat2 = Eigen::Matrix2cd;
using vec2 = Eigen::Vector2cd; 
using fieldMat = Eigen::Matrix<vec2, Nt, Nx>;
complex<double> const im {0, 1};

int const Nf = 2; // Number of flavours
int const vol = 2*Nx*Nt*Nf;
double const p=2.0*3.0*M_PI/Nt, q=2.0*6.0*M_PI/Nx; // useful to test Dirac operator

typedef class SpinorField {
    public: 
        int const Nt, Nx, Nf;
        SpinorField(int const Nt, int const Nx, int const Nf);
        ~SpinorField() {;}
        vector<complex<double>> val;
    private:
        random_device rnddev;
        mt19937 rndgen;
        normal_distribution<double> dist;
} SpinorField;

void DiracmatSpinorProduct(SpinorField const& inPsi, SpinorField& outPsi); // applies dirac operator to s
unsigned int PBC(int const n, int const N){return (n+N)%N;}
//unsigned int toFlat(int const nt, int const nx, int const f, int const c){return (c + 2*f + 2*nt*Nf + 2*nx*Nt*Nf);}
unsigned int toEOFlat(int const nt, int const nx, int const f, int const c);


int main(){

    ofstream datafile;  

    SpinorField psiField(Nt, Nx, Nf);
    SpinorField psiField2(Nt, Nx, Nf);
    DiracmatSpinorProduct(psiField, psiField2);
    
    // Test Dirac operator (see notes for details on what is going on here)
    /*complex<double> dcp[2];
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            for(int f=0; f<Nf; f++){
                dcp[0] = (2.0-cos(p)-cos(q)+im*sin(p))*psiField.val[toFlat(nt, nx, f, 0)] + im*sin(q)*psiField.val[toFlat(nt, nx, f, 1)];
                dcp[1] = (2.0-cos(p)-cos(q)-im*sin(p))*psiField.val[toFlat(nt, nx, f, 1)] - im*sin(q)*psiField.val[toFlat(nt, nx, f, 0)];
                cout << psiField2.val[toFlat(nt, nx, f, 0)] << " " << dcp[0] << endl;
                cout << psiField2.val[toFlat(nt, nx, f, 1)] << " " << dcp[1] << endl;
            }
            cout << endl;
        }
    }*/



    return 0;
}

unsigned int toEOFlat(int const nt, int const nx, int const f, int const c){
    int const s = Nt*Nx/2*Nf*2;
    if ((nt+nx) % 2 == 0) return  nt*Nx/2*Nf*2 + nx/2*Nf*2 + f*2 + c;
    else return s + nt*Nx/2*Nf + nx/2*Nf + f*2 + c;
}


SpinorField::SpinorField(int const Nt, int const Nx, int const Nf) :
    rndgen(rnddev()),
    dist(0., 1.),
    Nt{Nt},
    Nx{Nx},
    Nf{Nf},
    val(Nt*Nx*Nf*2)
{
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            for(int f=0; f<Nf; f++){
                val[toEOFlat(nt, nx, f, 0)] = 1.0 * exp(im*p*(double)nt+im*q*(double)nx);
                val[toEOFlat(nt, nx, f, 1)] = 1.3 * exp(im*p*(double)nt+im*q*(double)nx);
            }
        }
    }
}


void DiracmatSpinorProduct(SpinorField const& inPsi, SpinorField& outPsi){
    assert(inPsi.Nt == outPsi.Nt && inPsi.Nx == outPsi.Nx && inPsi.Nf == outPsi.Nf);
    int Nt = inPsi.Nt, Nx = inPsi.Nx, Nf = inPsi.Nf;
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            for(int f=0; f<Nf; f++){
                outPsi.val[toEOFlat(nt, nx, f, 0)] =  2.0*inPsi.val[toEOFlat(nt, nx, f, 0)] 
                                                    - inPsi.val[toEOFlat(PBC(nt-1, Nt), nx, f, 0)] 
                                                    - 0.5*(inPsi.val[toEOFlat(nt, PBC(nx-1, Nx), f, 0)] + inPsi.val[toEOFlat(nt, PBC(nx-1, Nx), f, 1)]) 
                                                    - 0.5*(inPsi.val[toEOFlat(nt, PBC(nx+1, Nx), f, 0)] - inPsi.val[toEOFlat(nt, PBC(nx+1, Nx), f, 1)]);
                outPsi.val[toEOFlat(nt, nx, f, 1)] =  2.0*inPsi.val[toEOFlat(nt, nx, f, 1)]
                                                    - inPsi.val[toEOFlat(PBC(nt+1, Nt), nx, f, 1)] 
                                                    - 0.5*(inPsi.val[toEOFlat(nt, PBC(nx-1, Nx), f, 1)] - inPsi.val[toEOFlat(nt, PBC(nx-1, Nx), f, 0)]) 
                                                    - 0.5*(inPsi.val[toEOFlat(nt, PBC(nx+1, Nx), f, 0)] + inPsi.val[toEOFlat(nt, PBC(nx+1, Nx), f, 1)]);
            }
        }
    }
}








