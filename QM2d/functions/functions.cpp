#include "functions.h" 


mat_fc buildCompositeOP(mat const& flavourMat, mat const& spinorMat){
    mat_fc M;
    for(int f1=0; f1<2; f1++){
        for(int f2=0; f2<2; f2++){
            for(int c1=0; c1<2; c1++){
                for(int c2=0; c2<2; c2++){
                    M(2*f1+c1, 2*f2+c2) = flavourMat(f1, f2) * spinorMat(c1, c2);
                }
            }
        }
    }
    return M;
}
mat_fc_single buildCompositeOP_single(mat_single const& flavourMat, mat_single const& spinorMat){
    mat_fc_single M;
    for(int f1=0; f1<2; f1++){
        for(int f2=0; f2<2; f2++){
            for(int c1=0; c1<2; c1++){
                for(int c2=0; c2<2; c2++){
                    M(2*f1+c1, 2*f2+c2) = flavourMat(f1, f2) * spinorMat(c1, c2);
                }
            }
        }
    }
    return M;
}
std::complex<double> dotProduct(std::vector<vec_fc>::iterator v1begin, std::vector<vec_fc>::iterator v1end, std::vector<vec_fc>::iterator v2begin){
    std::complex<double> r = 0.0;
    for( ; v1begin != v1end; v1begin++, v2begin++) r += v1begin->dot(*v2begin);
    return r; 
}

std::complex<float> dotProduct(std::vector<vec_fc_single>::iterator v1begin, std::vector<vec_fc_single>::iterator v1end, std::vector<vec_fc_single>::iterator v2begin){
    std::complex<float> r = 0.0;
    for( ; v1begin != v1end; v1begin++, v2begin++) r += v1begin->dot(*v2begin);
    return r;
}


