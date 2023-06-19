#pragma once

#include <thrust/complex.h>


template <typename T>
struct Spinor {
    Spinor();
    thrust::complex<double> val[4];
};
