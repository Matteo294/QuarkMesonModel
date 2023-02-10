#include <vector>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>

class C{
    public:
        template <class T>
        void dot(T v1b, T v1e,T v2b){
            double s = 0.0;
            for(; v1b != v1e; v1b++, v2b++) s += (*v1b) * (*v2b);
            std::cout << s << std::endl;
        }
};


int main(){
    Eigen::Vector2d v1 {1.0, 1.0};
    C c;
    c.dot(v1.begin(), v1.end(), v1.begin());
    return 0;
}