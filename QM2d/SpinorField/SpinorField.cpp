#include "SpinorField.h"
#include <iostream>

SpinorField::SpinorField(Lattice& l) :
    rndgen(rnddev()),
    dist(0., 1.),
    l{l},
    val(l.Nt*l.Nx, vec_fc::Zero())
{
    val[0] = vec_fc {1.0, 1.0, 1.0, 1.0}; // source at the origin
}

SpinorField::SpinorField(SpinorField const& s) :
    rndgen(rnddev()),
    dist(0., 1.),
    l{s.l},
    val{s.val}
    {;}

void SpinorField::operator = (SpinorField const& s){
    assert(l.Nt == s.l.Nt && l.Nx == s.l.Nx);
    val = s.val;
}

std::complex<double> SpinorField::dot(SpinorField& s){
    std::complex<double> r = 0.0;
    for(int i=0; i<l.vol; i++) r += val[i].dot(s.val[i]);
    return r;
}
