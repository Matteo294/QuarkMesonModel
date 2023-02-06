#include "SpinorField.h"
#include <iostream>

SpinorField::SpinorField(int const vol) :
    rndgen(rnddev()),
    dist(0., 1.),
    vol{vol},
    val(vol, vec_fc::Zero())
{
    val[0] = vec_fc {1.0, 1.0, 1.0, 1.0}; // source at the origin
}

SpinorField::SpinorField(SpinorField const& s) :
    rndgen(rnddev()),
    dist(0., 1.),
    vol{s.vol},
    val{s.val}
    {;}

void SpinorField::operator = (SpinorField const& s){
    assert(vol == s.vol);
    val = s.val;
}
