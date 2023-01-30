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
