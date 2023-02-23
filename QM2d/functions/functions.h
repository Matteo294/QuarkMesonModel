#pragma once 

#include <complex>
#include <iostream>
#include <vector>

std::complex<double> constexpr im (0.0, 1.0);
std::complex<float> constexpr imf (0.0, 1.0);

// template can be float or double
template <typename T>
struct Spinor {
    Spinor() : val(4, 0.0) {};
    ~Spinor() {};
    void setZero() {for (auto& v: val) v = 0.0;}
    std::complex<T> dot(Spinor const& s) {
        std::complex<T> res = 0.0;
        for (int i=0; i<4; i++) res += conj(val[i]) * s.val[i];
        return res;
    }
    std::vector<std::complex<T>> val;
};

typedef Spinor<double> Spinor_d;
typedef Spinor<float> Spinor_f;

template <typename T>
struct O4Mat{
    O4Mat<T>() : val(2, std::vector<std::complex<T>> (2, 0.0)) {}
    O4Mat<T>(std::complex<T> const sigma, std::complex<T> const pi[3]) : val(2, std::vector<std::complex<T>> (2, 0.0)) {
        std::complex<T> im (0.0, 1.0);
        val[0][0] = sigma + im*pi[2]; 
        val[0][1] = im*(pi[0] - im*pi[1]);
        val[1][0] = im*(pi[0] + im*pi[1]);
        val[1][1] = sigma - im*pi[2];
    };
    std::complex<T> determinant() {return val[0][0]*val[1][1] - val[1][0]*val[0][1];}
    std::vector<std::vector<std::complex<T>>> val;
};

typedef O4Mat<double> O4Mat_d;
typedef O4Mat<float> O4Mat_f;

enum class MatrixType { Normal, Dagger };

using vecfield = std::vector<Spinor_d>;
using vecfield_iter = vecfield::iterator;
using vecfield_single = std::vector<Spinor_f>;
using vecfield_single_iter = vecfield_single::iterator;

using mat_single = std::vector<std::vector<std::complex<float>>>;


template <typename SpinorT>
std::complex<double> dotProduct(SpinorT v1begin, SpinorT v1end, SpinorT v2begin);

/*std::complex<double> dotProduct(vecfield_iter v1begin, vecfield_iter v1end, vecfield_iter v2begin);
std::complex<float> dotProduct(vecfield_single_iter v1begin, vecfield_single_iter v1end, vecfield_single_iter v2begin);*/
std::complex<double> dotProduct(std::vector<std::complex<double>>::iterator v1begin, std::vector<std::complex<double>>::iterator v1end, std::vector<std::complex<double>>::iterator v2begin);

void spinorSum(vecfield_iter s1begin, vecfield_iter s1end, vecfield_iter s2begin, vecfield_iter resbegin);
void spinorDiff(vecfield_iter s1begin, vecfield_iter s1end, vecfield_iter s2begin, vecfield_iter resbegin);

