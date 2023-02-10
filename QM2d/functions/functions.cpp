#include "functions.h" 

std::complex<double> dotProduct(spinor_iter v1begin, spinor_iter v1end, spinor_iter v2begin){
    std::complex<double> r = 0.0;
    for( ; v1begin != v1end; v1begin++, v2begin++) r += v1begin->dot(*v2begin);
    return r; 
}

std::complex<float> dotProduct(spinor_single_iter v1begin, spinor_single_iter v1end, spinor_single_iter v2begin){
    std::complex<float> r = 0.0;
    for( ; v1begin != v1end; v1begin++, v2begin++) r += v1begin->dot(*v2begin);
    return r;
}


