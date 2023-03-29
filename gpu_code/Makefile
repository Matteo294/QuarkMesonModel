#FLAGS= -rdc=true -arch=sm_61 -O2
#FLAGS= -rdc=true -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_62,code=sm_62 -gencode=arch=compute_70,code=sm_70 -O2
FLAGS= -rdc=true -gencode=arch=compute_70,code=sm_70 -O2

all: main
Lattice.o: Lattice.cu Lattice.cuh Makefile
	nvcc -c Lattice.cu -expt-relaxed-constexpr $(FLAGS)
Dirac.o: Dirac.cu  Dirac.cuh Makefile
	nvcc -c Dirac.cu  -expt-relaxed-constexpr $(FLAGS)

# Main function
main.o: main.cu Lattice.cuh Dirac.cuh Makefile 
	nvcc -c main.cu -expt-relaxed-constexpr $(FLAGS)
main: main.o Lattice.o Dirac.o Makefile
	nvcc -o main main.o Lattice.o Dirac.o $(FLAGS)

# Test Dirac operator
testD.o: tests/testD.cu Lattice.cuh Dirac.cuh Makefile
	nvcc -c tests/testD.cu -expt-relaxed-constexpr $(FLAGS)
testD: testD.o Lattice.o Dirac.o Makefile
	nvcc -o main testD.o Lattice.o Dirac.o $(FLAGS)

# Test inverse Dirac operator
testDinv.o: tests/testDinv.cu Lattice.cuh Dirac.cuh Makefile
	nvcc -c tests/testDinv.cu -expt-relaxed-constexpr $(FLAGS)
testDinv: testDinv.o Lattice.o Dirac.o Makefile
	nvcc -o main testDinv.o Lattice.o Dirac.o $(FLAGS)

# Test dot product
testDot.o: tests/testDot.cu Lattice.cuh Dirac.cuh Makefile
	nvcc -c tests/testDot.cu -expt-relaxed-constexpr $(FLAGS)
testDot: testDot.o Lattice.o Dirac.o Makefile
	nvcc -o main testDot.o Lattice.o Dirac.o $(FLAGS)

# Test drift
testDrift.o: tests/testDrift.cu Lattice.cuh Dirac.cuh Makefile
	nvcc -c tests/testDrift.cu -expt-relaxed-constexpr $(FLAGS)
testDrift: testDrift.o Lattice.o Dirac.o Makefile
	nvcc -o main testDrift.o Lattice.o Dirac.o $(FLAGS)

# Clean stuff
clean:
	rm -f *.o main *.csv *.out slurm* *.o* *.e*
