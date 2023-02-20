#include "functions.h" 

std::complex<double> dotProduct(vecfield_iter v1begin, vecfield_iter v1end, vecfield_iter v2begin){
    std::complex<double> r = 0.0;
    const int vol = std::distance(v1begin, v1end);
    for(int i=0; i<vol; i++) r += dotProduct(v1begin[i].val.begin(), v1begin[i].val.end(), v2begin[i].val.begin());
    return r; 
}

/*std::complex<float> dotProduct(vecfield_single_iter v1begin, vecfield_single_iter v1end, vecfield_single_iter v2begin){
    std::complex<float> r = 0.0;
    for( ; v1begin != v1end; v1begin++, v2begin++) r += v1begin->dot(*v2begin);
    return r;
}*/

std::complex<double> dotProduct(std::vector<std::complex<double>>::iterator v1begin, std::vector<std::complex<double>>::iterator v1end, std::vector<std::complex<double>>::iterator v2begin){
    std::complex<double> r = 0.0;
    const int vol = std::distance(v1begin, v1end);
    for(int i=0; i<vol; i++) r += conj(v1begin[i]) * v2begin[i];
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

O4Mat::O4Mat() : val(2, std::vector<std::complex<double>> (2, 0.0)) {};
O4Mat::O4Mat(std::complex<double> const sigma, std::complex<double> const pi[3]) : val(2, std::vector<std::complex<double>> (2)){
    val[0][0] = sigma + im*pi[2]; 
    val[0][1] = im*(pi[0] - im*pi[1]);
    val[1][0] = im*(pi[0] + im*pi[1]);
    val[1][1] = sigma - im*pi[2];
}
std::complex<double> O4Mat::determinant(){ return val[0][0]*val[1][1] - val[1][0]*val[0][1]; }

O4Mat_single::O4Mat_single() : val(2, std::vector<std::complex<float>> (2, 0.0)) {};
O4Mat_single::O4Mat_single(std::complex<float> sigma, std::complex<float> pi[3]) : val(2, std::vector<std::complex<float>> (2)){
    val[0][0] = (std::complex<float>) (sigma + imf*pi[2]); 
    val[0][1] = (std::complex<float>) (imf*(pi[0] - imf*pi[1]));
    val[1][0] = (std::complex<float>) (imf*(pi[0] + imf*pi[1]));
    val[1][1] = (std::complex<float>) (sigma - imf*pi[2]);
}