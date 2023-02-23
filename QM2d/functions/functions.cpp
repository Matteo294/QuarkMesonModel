#include "functions.h" 

/*std::complex<double> dotProduct(vecfield_iter v1begin, vecfield_iter v1end, vecfield_iter v2begin){
    std::complex<double> r = 0.0;
    const int vol = std::distance(v1begin, v1end);
    for(int i=0; i<vol; i++) r += dotProduct(v1begin[i].val.begin(), v1begin[i].val.end(), v2begin[i].val.begin());
    return r; 
}*/

/*std::complex<float> dotProduct(vecfield_single_iter v1begin, vecfield_single_iter v1end, vecfield_single_iter v2begin){
    std::complex<float> r = 0.0;
    for( ; v1begin != v1end; v1begin++, v2begin++) r += v1begin->dot(*v2begin);
    return r;
}*/


template <typename SpinorT>
std::complex<double> dotProduct(SpinorT v1begin, SpinorT v1end, SpinorT v2begin){
    std::complex<double> r = 0.0;
    const int vol = std::distance(v1begin, v1end);
    for(int i=0; i<vol; i++) r += v1begin[i].dot(v2begin[i]);
    return r; 
}

void spinorSum(vecfield_iter s1begin, vecfield_iter s1end, vecfield_iter s2begin, vecfield_iter resbegin){
    const int vol = std::distance(s1begin, s1end);
    for(int i=0; i<vol; i++){
        for(int j=0; j<4; j++) resbegin[i].val[j] = s1begin[i].val[j] + s2begin[i].val[j];
    }
}

void spinorDiff(vecfield_iter s1begin, vecfield_iter s1end, vecfield_iter s2begin, vecfield_iter resbegin){
    const int vol = std::distance(s1begin, s1end);
    for(int i=0; i<vol; i++){
        for(int j=0; j<4; j++) resbegin[i].val[j] = s1begin[i].val[j] - s2begin[i].val[j];
    }
}

template std::complex<double> dotProduct<vecfield_iter>(vecfield_iter v1begin, vecfield_iter v1end, vecfield_iter v2begin);