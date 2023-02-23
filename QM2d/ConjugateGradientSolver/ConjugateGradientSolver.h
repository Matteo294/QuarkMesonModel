#pragma once
#include "../SpinorField/SpinorField.h"
#include "../DiracOP/DiracOP.h"
#include "../functions/functions.h"
#include <algorithm>


template <typename Titer, typename Tvar>
class ConjugateGradientSolver{
    public:
        ConjugateGradientSolver(int const IterMax, double const tolerance, DiracOP<Titer, Tvar>& Dirac);
        ~ConjugateGradientSolver(){}
        void solve_D(Titer ybegin, Titer yend, Titer xbegin);
        void solve_Dhat(Titer ybegin, Titer yend, Titer xbegin);
    protected:
        double const tolerance;
        int const IterMax;
        DiracOP<Titer, Tvar>& Dirac;
};

template <typename Titer, typename Tvar>
ConjugateGradientSolver<Titer, Tvar>::ConjugateGradientSolver(int const IterMax, double const tolerance, DiracOP<Titer, Tvar>& Dirac) :
    IterMax{IterMax},
    tolerance{tolerance},
    Dirac{Dirac}
    {;}

template <typename Titer, typename Tvar>
void ConjugateGradientSolver<Titer, Tvar>::solve_D(Titer ybegin, Titer yend, Titer xbegin){
    const int vol = std::distance(ybegin, yend);
    
    std::vector<Spinor<Tvar>> r(vol, Spinor<Tvar>()), p(vol, Spinor<Tvar>()), temp(vol, Spinor<Tvar>()), temp2(vol, Spinor<Tvar>());
    std::complex<Tvar> alpha;
    Tvar beta, rmodsq;
    
    std::fill(xbegin, xbegin + vol, Spinor<Tvar>());

    std::copy(ybegin, yend, r.begin());
    p = r;
    rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();

    int k;
    for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

        Dirac.applyTo(p.begin(), temp2.begin(), MatrixType::Dagger);
        Dirac.applyTo(temp2.begin(), temp.begin(), MatrixType::Normal);

        alpha = rmodsq / (dotProduct(p.begin(), p.end(), temp.begin())).real(); 
        // x += alpha p
        for(int i=0; i<vol; i++) std::transform(p[i].val.begin(), p[i].val.end(), temp2[i].val.begin(), [alpha](auto& c){return c*alpha;});
        spinorSum<Titer>(temp2.begin(), temp2.end(), xbegin, xbegin);
        // r -= alpha A p
        for(int i=0; i<vol; i++) std::transform(temp[i].val.begin(), temp[i].val.end(), temp2[i].val.begin(), [alpha](auto& c){return c*alpha;});
        spinorDiff<Titer>(r.begin(), r.end(), temp2.begin(), r.begin());

        beta = (dotProduct(r.begin(), r.end(), r.begin())).real() / rmodsq;

        // p = r - beta p
        for(int i=0; i<vol; i++) std::transform(p[i].val.begin(), p[i].val.end(), temp2[i].val.begin(), [beta](auto& c){return c*beta;});
        spinorSum<Titer>(temp2.begin(), temp2.end(), r.begin(), p.begin());

        rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();
    }

    if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
    else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << rmodsq << "\n";

}

template <typename Titer, typename Tvar>
void ConjugateGradientSolver<Titer, Tvar>::solve_Dhat(Titer ybegin, Titer yend, Titer xbegin){
    const int vol = std::distance(ybegin, yend);
    
    std::vector<Spinor<Tvar>> r(vol, Spinor<Tvar>()), p(vol, Spinor<Tvar>()), temp(vol, Spinor<Tvar>()), temp2(vol, Spinor<Tvar>());
    std::complex<double> alpha;
    double beta, rmodsq;
    
    std::fill(xbegin, xbegin + vol, Spinor_d());

    std::copy(ybegin, yend, r.begin());
    p = r;
    rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();

    int k;
    for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

        Dirac.applyDhatTo(p.begin(), temp2.begin(), MatrixType::Dagger);
        Dirac.applyDhatTo(temp2.begin(), temp.begin(), MatrixType::Normal);

        alpha = rmodsq / (dotProduct(p.begin(), p.end(), temp.begin())).real(); 
        for(int i=0; i<vol; i++) std::transform(p[i].val.begin(), p[i].val.end(), temp2[i].val.begin(), [alpha](auto& c){return c*alpha;});

        spinorSum<Titer>(temp2.begin(), temp2.end(), xbegin, xbegin);

        for(int i=0; i<vol; i++) std::transform(temp[i].val.begin(), temp[i].val.end(), temp2[i].val.begin(), [alpha](auto& c){return c*alpha;});

        spinorDiff<Titer>(r.begin(), r.end(), temp2.begin(), r.begin());

        beta = (dotProduct(r.begin(), r.end(), r.begin())).real() / rmodsq;

        // p = r - beta p
        for(int i=0; i<vol; i++) std::transform(p[i].val.begin(), p[i].val.end(), temp2[i].val.begin(), [beta](auto& c){return c*beta;});
        spinorSum<Titer>(temp2.begin(), temp2.end(), r.begin(), p.begin());

        rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();
    }

    if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
    else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << rmodsq << "\n";

}

