#pragma once
#include <array>
#include <vector>
#include <algorithm>

#include "params.h"
#include "Lattice.cuh"

class Laplace {
public:
	Laplace(int const n);
	~Laplace();

	void func2();
	void func2_eo();

	int * const I;
	int * const J;
	myType * const cval;

private:
	int const N;
	std::vector<int> row_indices, col_indices;
	std::vector<myType> vals;
};
