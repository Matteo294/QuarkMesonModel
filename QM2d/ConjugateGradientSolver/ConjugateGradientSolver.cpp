#include "ConjugateGradientSolver.h"

ConjugateGradientSolver::ConjugateGradientSolver(int const IterMax, double const tolerance, DiracOP& Dirac) :
    IterMax{IterMax},
    tolerance{tolerance},
    Dirac{Dirac}
    {;}

void ConjugateGradientSolver::baseCG(SpinorField const& inPsi, SpinorField& outPsi){
    assert(inPsi.l.Nt == outPsi.l.Nt && inPsi.l.Nx == outPsi.l.Nx);
    
    SpinorField 
        r(inPsi.l), // residual
        p(inPsi.l), 
        temp(inPsi.l); // Store results of D Ddagger psi = D _temp
    std::complex<double> alpha;
    double beta, rmodsq;
    
    for(auto& v: outPsi.val) v.setZero();

    r.val = inPsi.val;
    p.val = r.val;
    rmodsq = (r.dot(r)).real();

    int k;
    for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

        temp = Dirac.applyTo(Dirac.applyTo(p, 1));

        alpha = rmodsq / (p.dot(temp)).real(); 
        for(int i=0; i<inPsi.l.vol; i++){
            outPsi.val[i] += alpha * p.val[i];
            r.val[i] -= alpha * temp.val[i]; 
        }
        beta = (r.dot(r)).real() / rmodsq;
        for(int i=0; i<inPsi.l.vol; i++) p.val[i] = r.val[i] + beta*p.val[i];
        rmodsq = (r.dot(r)).real();
    }

    if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
    else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << rmodsq << "\n";
}

void ConjugateGradientSolver::eoCG(SpinorField const& inPsi, SpinorField& outPsi){
    assert(inPsi.l.Nt == outPsi.l.Nt && inPsi.l.Nx == outPsi.l.Nx);
    
    SpinorField 
        r(inPsi.l), // residual
        p(inPsi.l), 
        temp(inPsi.l); // Store results of D Ddagger psi = D _temp
    std::complex<double> alpha;
    double beta, rmodsq;

    for(int i=0; i<outPsi.l.vol; i++) outPsi.val[i].setZero();
    
    for(auto& v: outPsi.val) v.setZero();
    r.val = inPsi.val;
    p.val = r.val;
    rmodsq = (r.dot(r)).real();

    int k;
    for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

        temp = Dirac.applyLDRTo(Dirac.applyLDRTo(p, 1));

        alpha = rmodsq / (p.dot(temp)).real(); 
        for(int i=0; i<inPsi.l.vol; i++){
            outPsi.val[i] += alpha * p.val[i];
            r.val[i] -= alpha * temp.val[i]; 
        }
        beta = (r.dot(r)).real() / rmodsq;
        for(int i=0; i<inPsi.l.vol; i++) p.val[i] = r.val[i] + beta*p.val[i];
        rmodsq = (r.dot(r)).real();
    }

    if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
    else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << rmodsq << "\n";
}