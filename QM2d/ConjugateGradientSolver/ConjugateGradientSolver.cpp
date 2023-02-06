#include "ConjugateGradientSolver.h"

ConjugateGradientSolver::ConjugateGradientSolver(int const IterMax, double const tolerance, DiracOP& Dirac) :
    IterMax{IterMax},
    tolerance{tolerance},
    Dirac{Dirac}
    {;}


void ConjugateGradientSolver::doubleCG(std::vector<vec_fc>::iterator ybegin, std::vector<vec_fc>::iterator yend, std::vector<vec_fc>::iterator xbegin){
    const int vol = Dirac.l.vol;
    
    std::vector<vec_fc> r(vol), p(vol), temp(vol), temp2(vol);
    std::complex<double> alpha;
    double beta, rmodsq;
    
    for(int i=0; i<vol; i++) xbegin[i].setZero();

    std::copy(ybegin, yend, r.begin());
    p = r;
    rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();

    int k;
    for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

        Dirac.applyTo(p.begin(), temp2.begin(), 1);
        Dirac.applyTo(temp2.begin(), temp.begin(), 0);

        alpha = rmodsq / (dotProduct(p.begin(), p.end(), temp.begin())).real(); 
        for(int i=0; i<vol; i++){
            xbegin[i] += alpha * p[i];
            r[i] -= alpha * temp[i]; 
        }
        beta = (dotProduct(r.begin(), r.end(), r.begin())).real() / rmodsq;
        for(int i=0; i<vol; i++) p[i] = r[i] + beta*p[i];
        rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();
    }

    if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
    else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << rmodsq << "\n";

}

void ConjugateGradientSolver::EOdoubleCG(std::vector<vec_fc>::iterator ybegin, std::vector<vec_fc>::iterator yend, std::vector<vec_fc>::iterator xbegin){
    const int vol = std::distance(ybegin, yend);
    
    std::vector<vec_fc> r(vol), p(vol), temp(vol), temp2(vol);
    std::complex<double> alpha;
    double beta, rmodsq;
    
    for(int i=0; i<vol; i++) xbegin[i].setZero();

    std::copy(ybegin, yend, r.begin());
    p = r;
    rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();

    int k;
    for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

        Dirac.applyDhatTo(p.begin(), temp2.begin(), 1);
        Dirac.applyDhatTo(temp2.begin(), temp.begin(), 0);

        alpha = rmodsq / (dotProduct(p.begin(), p.end(), temp.begin())).real(); 
        for(int i=0; i<vol; i++){
            xbegin[i] += alpha * p[i];
            r[i] -= alpha * temp[i]; 
        }
        beta = (dotProduct(r.begin(), r.end(), r.begin())).real() / rmodsq;
        for(int i=0; i<vol; i++) p[i] = r[i] + beta*p[i];
        rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();
    }

    if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
    else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << rmodsq << "\n";

}

void ConjugateGradientSolver::singleCG(std::vector<vec_fc_single>::iterator ybegin, std::vector<vec_fc_single>::iterator yend, std::vector<vec_fc_single>::iterator xbegin){
    const int vol = Dirac.l.vol;
    
    std::vector<vec_fc_single> r(vol), p(vol), temp(vol), temp2(vol);
    std::complex<float> alpha;
    float beta, rmodsq;
    
    for(int i=0; i<vol; i++) xbegin[i].setZero();

    std::copy(ybegin, yend, r.begin());
    p = r;
    rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();

    int k;
    for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

        Dirac.applyTo_single(p.begin(), temp2.begin(), 1);
        Dirac.applyTo_single(temp2.begin(), temp.begin(), 0);

        alpha = rmodsq / (dotProduct(p.begin(), p.end(), temp.begin())).real(); 
        for(int i=0; i<vol; i++){
            xbegin[i] += alpha * p[i];
            r[i] -= alpha * temp[i]; 
        }
        beta = (dotProduct(r.begin(), r.end(), r.begin())).real() / rmodsq;
        for(int i=0; i<vol; i++) p[i] = r[i] + beta*p[i];
        rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();
    }

    if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
    else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << rmodsq << "\n";
}