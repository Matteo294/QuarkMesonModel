#include "SpinorField.h"
#include <iostream>

SpinorField::SpinorField(int const Nt, int const Nx, int const Nf) :
    rndgen(rnddev()),
    dist(0., 1.),
    Nt{Nt},
    Nx{Nx},
    Nf{Nf},
    volume{Nt*Nx},
    val(Nt*Nx, vec_fc::Zero())
{
    val[0] = vec_fc {1.0, 1.0, 1.0, 1.0}; // source at the origin
}

SpinorField::SpinorField(SpinorField const& s) :
rndgen(rnddev()),
    dist(0., 1.),
    Nt{s.Nt},
    Nx{s.Nx},
    Nf{s.Nf},
    volume{s.volume},
    val{s.val}
        {;}

void SpinorField::operator = (SpinorField const& s){
    assert(Nt == s.Nt && Nx == s.Nx && Nf == s.Nf);
    val = s.val;
}

std::complex<double> SpinorField::dot(SpinorField& s){
    std::complex<double> r = 0.0;
    for(int i=0; i<volume; i++) r += val[i].dot(s.val[i]);
    return r;
}
