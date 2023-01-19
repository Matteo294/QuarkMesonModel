#include <iostream>
#include <cmath>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>


using namespace std;

int main(){
    Eigen::Matrix4d M {
        {6, 7, 0, 0},
        {1, 0, 3, 0},
        {0, 0, 3, 0},
        {1, 0, 0, 5}
    };
    Eigen::Vector<double, 7> v {6, 7, 1, 3, 3, 1, 5};
    Eigen::Vector<double, 7> COL_INDEX {0, 1, 0, 2, 2, 0, 3};
    Eigen::Vector<double, 5> ROW_INDEX {0, 2, 4, 5, 7};
    Eigen::Vector<double, 4> x {1, 2, 3, 4};

    Eigen::Vector<double, 4> y;
    for(int i=0; i<4; i++){
        y[i] = 0.0;
        for(int j=ROW_INDEX[i]; j<ROW_INDEX[i+1]; j++){
            y[i] += v[j] * x[COL_INDEX[j]];
        }
    }

    cout << y << endl << endl;
    cout << M*x << endl;


}