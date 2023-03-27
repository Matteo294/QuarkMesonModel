#pragma once

template <typename T>
struct Spinor {
    Spinor(){setZero();}
    __host__ __device__ void setZero() {for (int i=0; i<4; i++) val[i] = 0.0;}
    __host__ __device__ thrust::complex<double> dot(Spinor const& s) {
        thrust::complex<double> res = 0.0;
        for (int i=0; i<4; i++) res += conj(val[i]) * s.val[i];
        return res;
    }
    thrust::complex<double> val[4];
};
