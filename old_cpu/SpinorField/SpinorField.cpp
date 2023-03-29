#include "SpinorField.h"
#include <iostream>

SpinorField::SpinorField(int const vol) :
    rndgen(rnddev()),
    dist(0., 1.),
    vol{vol},
    pos(vol)
    {}

SpinorField::SpinorField(SpinorField const& s) :
    rndgen(rnddev()),
    dist(0., 1.),
    vol{s.vol},
    pos{s.pos}
    {;}

void SpinorField::operator = (SpinorField const& s){
    assert(vol == s.vol);
    pos = s.pos;
}