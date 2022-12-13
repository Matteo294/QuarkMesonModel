#include <iostream>
#include <vector>
#include <complex>
#include <random>
#include <fstream>
#include <cassert>
#include <algorithm>


/// !!!!!!!!!!!!!! Remember to add APBE
/// !!! check references when cycling thourgh vectors


int const Nt = 8, Nx = 8;

using namespace std;

complex<double> const im {0, 1};

int const Nf = 2; // Number of flavours
int const vol = 2*Nx*Nt*Nf;
double const p=2.0*3.0*M_PI/Nt, q=2.0*6.0*M_PI/Nx; // useful to test Dirac operator
// CG params
int const IterMax = 1000;
double const tol =1e-3;

typedef class SpinorField {
    public: 
        int const Nt, Nx, Nf, volume;
        SpinorField(int const Nt, int const Nx, int const Nf);
        ~SpinorField() {;}
        vector<complex<double>> val;
        complex<double> dot(SpinorField const& inPsi);
    private:
        random_device rnddev;
        mt19937 rndgen;
        normal_distribution<double> dist;
} SpinorField;

void DiracmatSpinorProduct(SpinorField const& inPsi, SpinorField& outPsi); // applies dirac operator to s
unsigned int PBC(int const n, int const N){return (n+N)%N;}
unsigned int toFlat(int const nt, int const nx, int const f, int const c){return (c + 2*f + 2*nt*Nf + 2*nx*Nt*Nf);}
unsigned int toEOFlat(int const nt, int const nx, int const f, int const c);

void CG(SpinorField const& inPsi, SpinorField& outPsi);



int main(){

    ofstream datafile;  

    SpinorField psiField(Nt, Nx, Nf);
    SpinorField psiField2(Nt, Nx, Nf);
    DiracmatSpinorProduct(psiField, psiField2);

    SpinorField afterCG(Nt, Nx, Nf);
    CG(psiField, afterCG);
    
    // Test Dirac operator (see notes for details on what is going on here)
    /*complex<double> dcp[2];
    for(int nt=0; nt<Nt; nt++){
        for(int nx=0; nx<Nx; nx++){
            for(int f=0; f<Nf; f++){
                dcp[0] = (2.0-cos(p)-cos(q) + im*sin(p))*psiField.val[toEOFlat(nt, nx, f, 0)] + im*sin(q)*psiField.val[toEOFlat(nt, nx, f, 1)];
                dcp[1] = (2.0-cos(p)-cos(q) - im*sin(p))*psiField.val[toEOFlat(nt, nx, f, 1)] + im*sin(q)*psiField.val[toEOFlat(nt, nx, f, 0)];
                cout << psiField2.val[toEOFlat(nt, nx, f, 0)] << " " << dcp[0] << endl;
                cout << psiField2.val[toEOFlat(nt, nx, f, 1)] << " " << dcp[1] << endl;
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
    volume{2*Nf*Nt*Nx},
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
                /*outPsi.val[toEOFlat(nt, nx, f, 0)] =  2.0*inPsi.val[toEOFlat(nt, nx, f, 0)] 
                                                    - inPsi.val[toEOFlat(PBC(nt-1, Nt), nx, f, 0)] 
                                                    - 0.5*(inPsi.val[toEOFlat(nt, PBC(nx-1, Nx), f, 0)] + inPsi.val[toEOFlat(nt, PBC(nx-1, Nx), f, 1)]) 
                                                    - 0.5*(inPsi.val[toEOFlat(nt, PBC(nx+1, Nx), f, 0)] - inPsi.val[toEOFlat(nt, PBC(nx+1, Nx), f, 1)]);
                outPsi.val[toEOFlat(nt, nx, f, 1)] =  2.0*inPsi.val[toEOFlat(nt, nx, f, 1)]
                                                    - inPsi.val[toEOFlat(PBC(nt+1, Nt), nx, f, 1)]  
                                                    - 0.5*(inPsi.val[toEOFlat(nt, PBC(nx-1, Nx), f, 0)] + inPsi.val[toEOFlat(nt, PBC(nx-1, Nx), f, 1)]) 
                                                    + 0.5*(inPsi.val[toEOFlat(nt, PBC(nx+1, Nx), f, 0)] - inPsi.val[toEOFlat(nt, PBC(nx+1, Nx), f, 1)]);*/
            }
        }
    }
    for(int i=0; i<inPsi.volume; i++) outPsi.val[i] = 2.0*inPsi.val[i];
}

complex<double> SpinorField::dot(SpinorField const& inPsi){
    assert(inPsi.Nt == Nt && inPsi.Nx == Nx && inPsi.Nf == Nf);
    complex<double> r = 0.0;
    for(int i=0; i<volume; i++) r += conj(val[i])*inPsi.val[i];
    return r;
}

void CG(SpinorField const& inPsi, SpinorField& outPsi){
    assert(inPsi.Nt == outPsi.Nt && inPsi.Nx == outPsi.Nx && inPsi.Nf == outPsi.Nf);
    SpinorField r(inPsi.Nt, inPsi.Nx, inPsi.Nf), rnew(inPsi.Nt, inPsi.Nx, inPsi.Nf), p(inPsi.Nt, inPsi.Nx, inPsi.Nf), temp(inPsi.Nt, inPsi.Nx, inPsi.Nf);

    r.val = inPsi.val;
    rnew.val = inPsi.val;
    for(auto& v: outPsi.val) v = 0.0;
    p.val = r.val;
    complex<double> alpha;
    double beta;

    for(int k=0; k<IterMax; k++){
        cout << k << " ";
        cout << sqrt(rnew.dot(rnew).real()) << endl;
        r.val = rnew.val;
        DiracmatSpinorProduct(inPsi, temp);
        alpha = r.dot(r) / p.dot(temp);
        for(int i=0; i<inPsi.volume; i++){
            outPsi.val[i] += alpha * p.val[k];
            rnew.val[i] = r.val[i] - alpha * temp.val[i]; 
        }
        if (sqrt(rnew.dot(rnew).real()) < tol) break;
        else{
            beta = rnew.dot(rnew).real() / r.dot(r).real();
            for(int i=0; i<inPsi.volume; i++) p.val[i] + rnew.val[i] + beta*p.val[i];
        }
    }
}







