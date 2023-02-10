#include "ConjugateGradientSolver.h"

ConjugateGradientSolver::ConjugateGradientSolver(int const IterMax, double const tolerance, DiracOP& Dirac) :
    IterMax{IterMax},
    tolerance{tolerance},
    Dirac{Dirac}
    {;}


void ConjugateGradientSolver::doubleCG_D(spinor_iter ybegin, spinor_iter yend, spinor_iter xbegin){
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

        Dirac.applyTo(p.begin(), temp2.begin(), MatrixType::Dagger);
        Dirac.applyTo(temp2.begin(), temp.begin(), MatrixType::Normal);

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

void ConjugateGradientSolver::doubleCG_Dhat(spinor_iter ybegin, spinor_iter yend, spinor_iter xbegin){
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

        Dirac.applyDhatTo(p.begin(), temp2.begin(), MatrixType::Dagger);
        Dirac.applyDhatTo(temp2.begin(), temp.begin(), MatrixType::Normal);

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

/*void ConjugateGradientSolver::singleCG(spinor_single_iter ybegin, spinor_single_iter yend, spinor_single_iter xbegin){
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

        Dirac.applyTo_single(p.begin(), temp2.begin(), MatrixType::Dagger);
        Dirac.applyTo_single(temp2.begin(), temp.begin(), MatrixType::Normal);

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
}*/

/*void ConjugateGradientSolver::mixedCG(spinor_iter ybegin, spinor_iter yend, spinor_iter xbegin, int const IterMaxSingle, double const toleranceSingle){

    const int vol = Dirac.l.vol;
    std::vector<vec_fc_single> r(vol), p(vol), temp(vol), temp2(vol), w(vol);
    std::vector<vec_fc> rd(vol), tempd(vol), temp2d(vol);
    std::vector<vec_fc_single> y(vol), x(vol);
    std::complex<float> alpha;
    float beta, rmodsq;
    double rmodsqd;

    for(int i=0; i<vol; i++) {y[i] = ybegin[i].cast<std::complex<float>>(); x[i] = xbegin[i].cast<std::complex<float>>();}
    for(int i=0; i<vol; i++) xbegin[i].setZero(); 

    std::copy(y.begin(), y.end(), r.begin());
    p = r;
    rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();
    rmodsqd = rmodsq;

    std::cout << rmodsqd << std::endl;
    
    for(int ktot=0; ktot<IterMax && sqrt(rmodsqd) > tolerance; ktot++){

        std::cout << ktot << std::endl;

        
        for(int k=0; k<IterMaxSingle && sqrt(rmodsq) > toleranceSingle; k++){

            std::cout << k << std::endl;

            Dirac.applyTo_single(p.begin(), temp2.begin(), MatrixType::Dagger);
            Dirac.applyTo_single(temp2.begin(), temp.begin(), MatrixType::Normal);

            alpha = rmodsq / (dotProduct(p.begin(), p.end(), temp.begin())).real(); 
            for(int i=0; i<vol; i++){
                x[i] += alpha * p[i];
                r[i] -= alpha * temp[i]; 
            }
            beta = (dotProduct(r.begin(), r.end(), r.begin())).real() / rmodsq;
            for(int i=0; i<vol; i++) p[i] = r[i] + beta*p[i];
            rmodsq = (dotProduct(r.begin(), r.end(), r.begin())).real();
        }

        for(int i=0; i<vol; i++) {xbegin[i] = x[i].cast<std::complex<double>>();}

        Dirac.applyTo(xbegin, temp2d.begin(), MatrixType::Dagger);
        Dirac.applyTo(temp2d.begin(), tempd.begin(), MatrixType::Normal);

        for(int i=0; i<vol; i++) {rd[i] = ybegin[i] - tempd[i];}

        Dirac.applyTo(xbegin, temp2d.begin(), MatrixType::Dagger);
        Dirac.applyTo(temp2d.begin(), tempd.begin(), MatrixType::Normal);

        for(int i=0; i<vol; i++) rd[i] = ybegin[i] - tempd[i];
        for(int i=0; i<vol; i++) r[i] = rd[i].cast<std::complex<float>>();

        for(int i=0; i<vol; i++) w[i] = p[i] - dotProduct(p.begin(), p.end(), r.begin())/dotProduct(r.begin(), r.end(), r.begin()) * r[i];

        Dirac.applyTo_single(w.begin(), temp2.begin(), MatrixType::Dagger);
        Dirac.applyTo_single(temp2.begin(), temp.begin(), MatrixType::Normal);

        for(int i=0; i<vol; i++) p[i] = r[i] - dotProduct(r.begin(), r.end(), temp2.begin())/dotProduct(w.begin(), w.end(), temp2.begin()) * r[i];
        rmodsqd = (dotProduct(rd.begin(), rd.end(), rd.begin())).real();
        
    }

}*/
