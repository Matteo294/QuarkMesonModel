#if !defined(USE_cuFFT) && !defined(USE_vkFFT)
#define USE_cUFFT
#endif

#include <cmath>
#include <atomic>
#include <vector>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <string>

#include <cooperative_groups.h>

#include "HDF5.h"
#include "toml.hpp"
#include "params.h"
#include "Laplace.h"
#include "device_info.h"
#include "managedVector.h"
#include "colouredNoise.h"
#include "reductions.cuh"
#include "langevin_gpu_v2.cuh"
#include "Dirac.cuh"
#include "Spinor.cuh"
#include "Lattice.cuh"

__constant__ myType epsBar;
__constant__ myType m2;
__constant__ myType lambda;

using std::conj;

thrust::complex<double> im {0.0, 1.0};


template <typename T>
__host__ void CGsolver_solve_D(Spinor<T> *inVec, Spinor<T> *outVec, DiracOP<T>& D, thrust::complex<double> *M, int const numBlocks, int const numThreads);

__global__ void gpuDotProduct(thrust::complex<double> *vecA, thrust::complex<double> *vecB, thrust::complex<double> *result, int size);

__host__ __device__ int EOtoNormal(int n, Lattice& l);
__host__ __device__ int NormalToEO(int n, Lattice& l);


void getForce(thrust::complex<double> *outVec, DiracOP<double>& D, thrust::complex<double> *M, Lattice& lattice, int const nBlocks_dot, int const nThreads_dot, int const nBlocks_drift, int const nThreads_drift);

__global__ void computeDrift(Spinor<double> *inVec, Spinor<double> *afterCG, thrust::complex<double> *outVec, int const vol);



double FindKappa(double const m2, double const lambda0) {
	auto const Delta = (2.0*nDim + m2)*(2.0*nDim + m2) + 4.0 * lambda0 / 3.0;
	auto const k1 = (-(2.0*nDim + m2) + std::sqrt(Delta)) / (2.0 * lambda0 / 3.0);

	return k1;
}

namespace {
	std::atomic<bool> early_finish = false;
}

void signal_handler(int signal) {
	early_finish = true;
}

int main(int argc, char** argv) {
	std::signal(SIGUSR2, signal_handler);





	// --------------------------------------------------------------
	double const fermion_mass = std::stod(argv[2]);
	
	thrust::complex<double> *M, *fermionic_contribution;
	Spinor<double> *in, *out;
	Lattice lattice(Nt, Nx);
	DiracOP<double> Dirac(fermion_mass, g_coupling, lattice);
	// Allocate two vectors and mesons matrix
	cudaMallocManaged(&M, sizeof(thrust::complex<double>) * 4 * lattice.vol);
	cudaMallocManaged(&in, sizeof(Spinor<double>) * lattice.vol);
	cudaMallocManaged(&out, sizeof(Spinor<double>) * lattice.vol);
	cudaMallocManaged(&fermionic_contribution, sizeof(thrust::complex<double>) * 4 * lattice.vol);

	int nBlocks_dot = 0, nBlocks_drift = 0;
	int nThreads_dot = 0, nThreads_drift = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks_dot, &nThreads_dot, gpuDotProduct);
	cudaDeviceSynchronize();
	cudaOccupancyMaxPotentialBlockSize(&nBlocks_drift, &nThreads_drift, computeDrift);
	cudaDeviceSynchronize();

	MatrixType useDagger = MatrixType::Normal;
	// diagArgs should be passed to all the diagonal (in spacetime) functions: Doo, Dee, Dooinv, Deeinv
	void *diagArgs[] = {(void*)&in, (void*)&out, (void*) &lattice.vol, (void*) &fermion_mass, (void*) &g_coupling, (void*)&useDagger, (void*)&M};
	// hopping should be passed to all the off-diagonal (in spacetime) functions: Deo, Doe
	void *hoppingArgs[] = {(void*)&in, (void*) &out, (void*) &lattice.vol, (void*) &useDagger, (void*) &lattice.IUP, (void*) &lattice.IDN}; 
	
	int numBlocks = 0;
	int numThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&numBlocks, &numThreads, gpuDotProduct);
	cudaDeviceSynchronize();

	std::ofstream datafile, tracefile;
	datafile.open("data.csv");
	datafile << "f0c0,f0c1,f1c0,f1c1" << "\n";
	std::string fname;
	fname.append("traces"); fname.append(argv[2]); fname.append(".csv");
	tracefile.open(fname);
	tracefile << "tr,sigma" << "\n";
	// --------------------------------------------------------------


	if constexpr(nDim > 3)
		std::cout << "#Due do technical limitations, coloured noise is *DISABLED* for nDim > 3.\n\n";

	if (argc == 1) {
		std::cerr << "No input file provided.\nExiting.\n";
		exit(1);
	}

	int constexpr N = SIZE;
	auto lap = Laplace{N};
	lap.func2();

	auto ivec  = ManagedVector<myType>{N * nVectorComponents};
	for (auto &e : ivec)
		e = static_cast<myType>(1.0 - 2.0*drand48());
//		e = static_cast<myType>(drand48());

	auto drift = ManagedVector<myType>{N * nVectorComponents};
	auto noise = ManagedVector<myType>{N * nVectorComponents};

	auto avg = ManagedVector<myType>{nVectorComponents};
	// timeSlices is organised with the field component as the *outer* index, and the time
	// coordinate as the inner index
	auto timeSlices = ManagedVector<myType>{nVectorComponents * Sizes[0]};

	// print out the parameters from the input file
	auto inFile = std::ifstream(argv[1]);
	std::string line;
	while (getline(inFile, line)) 
		std::cout << '#' << line << '\n';
	inFile.close();
	std::cout << "\n\n";

	auto const inputData = toml::parse(argv[1]);
	auto const& parameters = toml::find(inputData, "physics");
	auto const useMass = toml::find<bool>(parameters, "useMass");
	myType my_m2, myLambda, kappa, Lambda;
	if (useMass == true) {
		my_m2 = toml::find<myType>(parameters, "mass");
		myLambda = toml::find<myType>(parameters, "g");

		kappa = FindKappa(my_m2, myLambda);
		Lambda = kappa*kappa*myLambda/6.0;
	} else {
		kappa  = toml::find<myType>(parameters, "kappa");
		Lambda = toml::find<myType>(parameters, "lambda");

		my_m2 = (1.0 - 2.0*Lambda) / kappa - 2.0*nDim;
		myLambda = 6.0 * Lambda / (kappa*kappa);
	}
	auto const sq2Kappa = std::sqrt(2.0 * kappa);
	auto const cutFraction = toml::find<myType>(parameters, "cutFraction");

	auto const& rndSection = toml::find(inputData, "random");
	int const seed = toml::find<int>(rndSection, "seed");

	auto const& langevin = toml::find(inputData, "langevin");
	auto	   myEpsBar = toml::find<double>(langevin, "averageEpsilon");
	auto const MaxLangevinTime = toml::find<double>(langevin, "MaxLangevinTime");
	auto const ExportTime = toml::find<double>(langevin, "ExportTime");
	auto const burnCount = toml::find<int>(langevin, "burnCount");
	auto const MeasureDriftCount = toml::find<int>(langevin, "MeasureDriftCount");

	auto const& ioSection = toml::find(inputData, "io");
	auto const outFileName = toml::find<std::string>(ioSection, "configFileName");
	auto const timeSliceFileName = toml::find<std::string>(ioSection, "timeSliceFileName");
	std::string startFileName = "";
	try {
		startFileName = toml::find<std::string>(ioSection, "startFileName");
	} catch (std::exception& e) {}
	bool exportHDF = false;
	try {
		exportHDF   = toml::find<bool>(ioSection, "export");
	} catch (std::exception& e) {}
	bool resumeRun = false;
	try {
		resumeRun = toml::find<bool>(ioSection, "resume");
	} catch (std::exception& e) {}

	auto hdf = HDF{outFileName, resumeRun};
	hdf.Close();

	if (resumeRun == true) {
		hdf.Open();
		if (hdf.NumberOfConfigs() == -1) {	// if there are no configs in the file, we cannot
			resumeRun = false;				// read from it
			std::cout << "#No configurations found in the HDF file.\n#Not resuming.\n";
		}
		hdf.Close();
	}

	if (resumeRun == false) {
		hdf.Open();
		hdf.CreateGroup("/seeds");std::ofstream datafile;
		datafile.open("data.csv");
		datafile << "f0c0,f0c1,f1c0,f1c1" << "\n";
		hdf.CreateGroup("/params");
		hdf.CreateGroup("/params/raw");
		hdf.WriteH5_Attribute("/params/raw/", "mass", my_m2);
		hdf.WriteH5_Attribute("/params/raw/", "g", myLambda);
//		hdf.WriteH5_Attribute("/params/raw/", "external_field0", external_field);
//		hdf.WriteH5_Attribute("/params/raw/", "shift", phi.GetShift() * sqrt(2.0*phi.Kappa()));

		hdf.CreateGroup("/params/dimensionless");
		hdf.WriteH5_Attribute("/params/dimensionless/", "kappa", kappa);
		hdf.WriteH5_Attribute("/params/dimensionless/", "lambda", Lambda);
//		hdf.WriteH5_Attribute("/params/dimensionless/", "external_field",
//				external_field / sqrt(2.0 * phi.Kappa()));
//		hdf.WriteH5_Attribute("/params/dimensionless/", "shift", phi.GetShift());

		hdf.CreateGroup("/data");
		hdf.Close();
	}
	//

	myType *maxDrift;
	myType *eps, elapsedLangevinTime;
	cudaMallocManaged(&maxDrift, sizeof(myType));
	cudaMalloc(&eps, sizeof(myType));
	myType *h_eps;
	h_eps = (myType*)malloc(sizeof(myType));
	//*eps = myEpsBar;
	cudaMemcpy(eps, &myEpsBar, sizeof(myType), cudaMemcpyHostToDevice);
	elapsedLangevinTime = 0.0;

	auto const kli = KernelLaunchInfo{Run};
	auto const kli_sMem = sizeof(myType) * std::max(kli.numThreads, 32);

	auto cn = ColouredNoise{noise, N, cutFraction, seed, kli};

	void *kMagnetisation[] = {(void*)&ivec, (void*)&(avg.data()), (void*)&N};

	auto const kTimeSlices = KernelLaunchInfo{gpuTimeSlices};
	void *timeSlicesArgs[] = {(void*)&ivec, (void*)&(timeSlices.data()), (void*)&N};

	// can't pass lap directly because it's not allocated on the device
	// is it worth it to allocate it on the device...? I, J, and cval are already there...
	// the only difference would be reducing the number of arguments here...
	void *kAll[] = {
		(void*)&eps,
		(void*)&ExportTime,
		(void*)&(ivec.data()),
		(void*)&(drift.data()),
		(void*)&(noise.data()),
		(void*)&N,
		(void*)&(lap.I),
		(void*)&(lap.J),
		(void*)&(lap.cval),
		(void*)&maxDrift,
		(void*)&fermionic_contribution};


	std::cout << std::endl;		// force a flush so we can see something on the screen before
								// actual computations start

	//

	cudaMemcpyToSymbol(m2, &my_m2, sizeof(myType));
	cudaMemcpyToSymbol(lambda, &myLambda, sizeof(myType));
	cudaMemcpyToSymbol(epsBar, &myEpsBar, sizeof(myType));
	// burn in a little bit, since the drift might be stronger at the beginning, since we are
	// likely far from the equilibrium state
	for (int burn = 0; burn < burnCount; ++burn) {
		myType t = 0.0;
		while (t < ExportTime) {
			cn();
			kli.Run(kAll, kli_sMem);
			cudaDeviceSynchronize();
			cudaMemcpy(h_eps, eps, sizeof(myType), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t += *h_eps;
		}
	}

	int nMeasurements = 0;
	int oldMeasurements = 0;
	elapsedLangevinTime = 0.0;
	if (resumeRun == true) {
		hdf.Open();
		auto configName = hdf.NameOfLastConfig();
//		std::cout << "name of last config " << configName << '\n';
		hdf.ReadData("/data/", configName + "/fields", ivec);
		hdf.ReadSeeds("/seeds/", "last", cn.GetState());
		configName.erase(0, configName.find('_') + 1);
		oldMeasurements = std::stoi(configName);
		hdf.Close();
	}

	if (MeasureDriftCount > 0) {
	myType epsSum = 0.0;
	while (elapsedLangevinTime < MeasureDriftCount * ExportTime) {
		myType t = 0.0;
		while (t < ExportTime) {
			cn();
			kli.Run(kAll, kli_sMem);
			cudaDeviceSynchronize();
			cudaMemcpy(h_eps, eps, sizeof(myType), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t += *h_eps;
		}
		elapsedLangevinTime += t;

		epsSum += *h_eps;
		nMeasurements++;
	}
	epsSum /= nMeasurements;
	std::cout << "#Average eps during drift measurement = " << epsSum << std::endl;
	myEpsBar *= myEpsBar/epsSum;	// update epsBar so that the average step size is roughly the
									// original value of epsBar provinded in the input file
	}
	cudaMemcpyToSymbol(epsBar, &myEpsBar, sizeof(myType));

	// main loop
	elapsedLangevinTime = 0.0;
	nMeasurements = oldMeasurements;
	std::vector<myType> hostLattice(N*nVectorComponents);
	elapsedLangevinTime = nMeasurements * ExportTime;
	auto timeSliceFile = std::ofstream(timeSliceFileName);

	thrust::complex<double> tr; // trace D^-1
	std::vector<thrust::complex<double>> traces;

	auto timerStart = std::chrono::high_resolution_clock::now();
	while (elapsedLangevinTime < MaxLangevinTime) {
		myType t = 0.0;
		while (t < ExportTime) {
			cn();
			
			// ------------------------------------------------------------------------------------------------
			for(int i=0; i<lattice.vol; i++){
				M[NormalToEO(i, lattice)] = ivec[i] + im * ivec[3*lattice.vol+i];
				M[NormalToEO(i + 3*lattice.vol, lattice)] = ivec[i] - im * ivec[3*lattice.vol+i];
				M[NormalToEO(i + lattice.vol, lattice)] = im * (ivec[lattice.vol+i] - im * ivec[2*lattice.vol+i]);
				M[NormalToEO(i + 2*lattice.vol, lattice)] = im * (ivec[lattice.vol+i] + im * ivec[2*lattice.vol+i]);
				fermionic_contribution[i] = 0.0;
				fermionic_contribution[i + lattice.vol] = 0.0;
				fermionic_contribution[i + 2 * lattice.vol] = 0.0;
				fermionic_contribution[i + 3 * lattice.vol] = 0.0;
			}
			getForce(fermionic_contribution, Dirac, M, lattice, nBlocks_dot, nThreads_dot, nBlocks_drift, nThreads_drift);
			std::cout << "force: " << fermionic_contribution[0] << " " << fermionic_contribution[lattice.vol] << " " << fermionic_contribution[2*lattice.vol] << " " << fermionic_contribution[3*lattice.vol] << "\n";
			for(int i=0; i<4*lattice.vol; i++){drift[i] = fermionic_contribution[i].real();}
			// ------------------------------------------------------------------------------------------------
			
			kli.Run(kAll, kli_sMem);
			cudaDeviceSynchronize();
			cudaMemcpy(h_eps, eps, sizeof(myType), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t += *h_eps;
		}
		
		elapsedLangevinTime += t;

//		cudaMemPrefetchAsync(ivec.data(), N*nVectorComponents, cudaCpuDeviceId);
		cudaLaunchCooperativeKernel((void*)gpuMagnetisation, kli.dimGrid, kli.dimBlock,
				kMagnetisation, kli_sMem, NULL);
		cudaDeviceSynchronize();


		kTimeSlices.Run(timeSlicesArgs, kli_sMem);
		cudaDeviceSynchronize();


		// -----------------------------------------------------------------------------
		// Set fields values
		for(int i=0; i<lattice.vol; i++){
			M[NormalToEO(i, lattice)] = ivec[i] + im * ivec[3*lattice.vol+i];
			M[NormalToEO(i + 3*lattice.vol, lattice)] = ivec[i] - im * ivec[3*lattice.vol+i];
			M[NormalToEO(i + lattice.vol, lattice)] = im * (ivec[lattice.vol+i] - im * ivec[2*lattice.vol+i]);
			M[NormalToEO(i + 2*lattice.vol, lattice)] = im * (ivec[lattice.vol+i] + im * ivec[2*lattice.vol+i]);
		}
		for(int i=0; i<lattice.vol; i++){in[i].setZero(); out[i].setZero();}
		// set source
		in[0].val[0] = 1.0;
		in[0].val[1] = 1.0;
		in[0].val[2] = 1.0;
		in[0].val[3] = 1.0;

		useDagger = MatrixType::Normal;
		diagArgs[0] = (void*) &in; diagArgs[1] = (void*) &out;
		hoppingArgs[0] = (void*) &in; hoppingArgs[1] = (void*) &out;
		CGsolver_solve_D(in, out, Dirac, M, numBlocks, numThreads);

		for(int i=0; i<lattice.vol; i++){in[i].setZero();}
		useDagger = MatrixType::Dagger;
		diagArgs[0] = (void*) &out; diagArgs[1] = (void*) &in;
		hoppingArgs[0] = (void*) &out; hoppingArgs[1] = (void*) &in;
		Dirac.applyD(diagArgs, hoppingArgs);


		thrust::complex<double> corr = 0.0;
		for(int nt=0; nt<Nt; nt++){
			corr = 0.0;
			for(int nx=0; nx<Nx; nx++){
				for(int j=0; j<4; j++) corr += in[lattice.toEOflat(nt, nx)].val[j];
			}
			datafile << corr.real() << "\n";
		}
		// -----------------------------------------------------------------------------







		for (int comp = 0; comp < nVectorComponents; ++comp) {
			for (int tt = 0; tt < nTimeSlices; ++tt)
				timeSliceFile << timeSlices[tt + nTimeSlices * comp] / SpatialVolume << '\t';
			timeSliceFile << '\n';
		}
		timeSliceFile << '\n';

		std::cout << elapsedLangevinTime << '\t' << *h_eps << '\t';
		myType sum2 = 0.0;
		for (auto e : avg) {
			if (useMass == false) e /= sq2Kappa;
			std::cout << e / N << '\t';
			sum2 += e*e;
		}
		std::cout << std::sqrt(sum2) / N << std::endl;

		tr = 0.0;
		for(int i=0; i<lattice.vol; i++) tr += fermionic_contribution[i];
		tr /= g_coupling;
		tracefile << tr.real()/(lattice.Nt*lattice.Nx) << "," << avg[0] /(lattice.Nt*lattice.Nx) << "\n";
		std::cout << "Trace: " << tr.real()/(lattice.Nt*lattice.Nx) << "\n";

		nMeasurements++;
		
		// this explicit copy seems to peform slightly/marginally better
		// TODO: needs further investigation
		cudaMemcpy(hostLattice.data(), ivec.data(), N*nVectorComponents*sizeof(myType),
				cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		// if the user provided kappa as input, we rescale the output field to dimensionless format
		if (useMass == false)
			for (auto& e : hostLattice)
				// divide or multiply...?
				e /= sq2Kappa;
		if (early_finish == true) {
			std::cout << "#Early termination signal received.\n#Wrapping up.\n";
			elapsedLangevinTime = MaxLangevinTime + 1.0;
		}
		std::stringstream ss;
		ss << "data/cnfg_" << std::setfill('0') << std::setw(8) << 
			(exportHDF == true ? nMeasurements : 1);
		hdf.Open();
		hdf.CreateGroup(ss.str());
		hdf.WriteData(ss.str(), "fields", hostLattice);
		hdf.Close();
		std::cout << "\n";
	}
	auto timerStop  = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart);
	timeSliceFile.close();
	hdf.Open();
	hdf.WriteSeeds("/seeds", "last", cn.GetState());
	hdf.Close();

	std::cout << "#numSms = " << kli.numSms << '\n';
	std::cout << "#blocks per SM = " << kli.numBlocksPerSm << '\n';
	std::cout << "#theads = " << kli.numThreads << '\n';
	std::cout << "#blocks = " << kli.numBlocks << '\n';

	std::cout << "#Number of measurements: " << nMeasurements << '\n';

	std::cout << "#Run time for main loop: " << duration.count() / 1000.0 << "s\n";

	cudaFree(eps);
	cudaFree(maxDrift);
	free(h_eps);

	// ---------------------------------
	cudaFree(M);
	cudaFree(in);
	cudaFree(out);
	cudaFree(&fermionic_contribution);
	// ---------------------------------

	return 0;
}

template <typename T>
__host__ void CGsolver_solve_D(Spinor<T> *inVec, Spinor<T> *outVec, DiracOP<T>& D, thrust::complex<double> *M, int const numBlocks, int const numThreads){	
	
	int const vol = D.lattice.vol;
	int mySize = D.lattice.vol * 4;

	Spinor<T> *r, *p, *temp, *temp2; // allocate space ?? 
	thrust::complex<T> alpha; // allocate space ??
	T beta, rmodsq;
	thrust::complex<double> *dot_res;

	cudaMallocManaged(&r, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&p, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&temp, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&temp2, sizeof(Spinor<T>) * vol);
	cudaMallocManaged(&dot_res, sizeof(thrust::complex<double>));

	for(int i=0; i<vol; i++) {
		outVec[i] = Spinor<T> ();
		temp[i] = Spinor<T> ();
		temp2[i] = Spinor<T> ();
		for(int j=0; j<4; j++) r[i].val[j] = inVec[i].val[j];
		for(int j=0; j<4; j++) p[i].val[j] = r[i].val[j];
	}

	// Set up dot product call
	void *dotArgs[] = {(void*) &r, (void*) &r, (void*) &dot_res, (void*) &mySize};
	auto dimGrid = dim3(numBlocks, 1, 1);
	auto dimBlock = dim3(numThreads, 1, 1);

	*dot_res = 0.0;
	cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
	cudaDeviceSynchronize();
	rmodsq = dot_res->real();

	MatrixType dag = MatrixType::Normal;

	void *diagArgs[] = {(void*)&p, (void*)&temp2, (void*) &D.lattice.vol, (void*) &D.fermion_mass, (void*) &g_coupling, (void*)&dag, (void*)&M};
	void *hoppingArgs[] = {(void*)&p, (void*)&temp2, (void*) &D.lattice.vol, (void*)&dag, (void*)&D.lattice.IUP, (void*)&D.lattice.IDN};

	int k;
	for(k=0; k<IterMax && sqrt(rmodsq) > tolerance; k++){

		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) temp[i].val[j] = 2.0 * p[i].val[j];
		}

		// Set buffers to zero to store the result fo the Dirac operator applied to p
		for(int i=0; i<D.lattice.vol; i++) {temp2[i].setZero(); temp[i].setZero();}

		// Apply D dagger
		dag = MatrixType::Dagger;
		diagArgs[0] = (void*) &p; diagArgs[1] = (void*) &temp2;
		hoppingArgs[0] = (void*) &p; hoppingArgs[1] = (void*) &temp2;
		D.applyD(diagArgs, hoppingArgs);
		// Apply D
		dag = MatrixType::Normal;
		diagArgs[0] = (void*) &temp2; diagArgs[1] = (void*) &temp;
		hoppingArgs[0] = (void*) &temp2; hoppingArgs[1] = (void*) &temp;
		D.applyD(diagArgs, hoppingArgs);
		
		dotArgs[0] = (void*) &p; dotArgs[1] = (void*) &temp;

		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
		cudaDeviceSynchronize();
		alpha = rmodsq / *dot_res; 

		// x = x + alpha p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) outVec[i].val[j] += alpha*p[i].val[j];
		}
		// r = r - alpha A p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) r[i].val[j] -= alpha*temp[i].val[j];
		}

		dotArgs[0] = (void*) &r; dotArgs[1] = (void*) &r;
		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
		cudaDeviceSynchronize();
		beta = abs(*dot_res) / rmodsq;

		// p = r - beta p
		for(int i=0; i<vol; i++){
			for(int j=0; j<4; j++) p[i].val[j] = r[i].val[j] + beta*p[i].val[j];
		}

		*dot_res = 0.0;
		cudaLaunchCooperativeKernel((void*)&gpuDotProduct, dimGrid, dimBlock, dotArgs, sizeof(thrust::complex<double>) * (32), NULL);
		cudaDeviceSynchronize();
		rmodsq = abs(*dot_res);
	}

	//if (k < IterMax) std::cout << "Convergence reached in " << k-1 << " steps \n";
	//else std::cout << "Max. number of iterations reached (" << IterMax << "), final err: " << sqrt(rmodsq) << "\n";

	cudaFree(r);
	cudaFree(p);
	cudaFree(temp);
	cudaFree(temp2);
	cudaFree(dot_res);
}


__global__ void gpuDotProduct(thrust::complex<double> *vecA, thrust::complex<double> *vecB, thrust::complex<double> *result, int size) {
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();
	//*result = 0.0;
	extern __shared__ thrust::complex<double> tmp[];

	thrust::complex<double> temp_sum = 0.0;
	for (int i = grid.thread_rank(); i < size; i += grid.size()) {
		temp_sum += conj(vecA[i]) * vecB[i];
	}

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	temp_sum = cg::reduce(tile32, temp_sum, cg::plus<thrust::complex<double>>());

	if (tile32.thread_rank() == 0) {
		tmp[tile32.meta_group_rank()] = temp_sum;
	}

	cg::sync(cta);

	if (tile32.meta_group_rank() == 0) {
		temp_sum = tile32.thread_rank() < tile32.meta_group_size() ? tmp[tile32.thread_rank()] : 0.0;
		temp_sum = cg::reduce(tile32, temp_sum, cg::plus<thrust::complex<double>>());

		if (tile32.thread_rank() == 0) {
		atomicAdd(reinterpret_cast<double*>(result), temp_sum.real());
		atomicAdd(reinterpret_cast<double*>(result)+1, temp_sum.imag());
		}
	}
}


void getForce(thrust::complex<double> *outVec, DiracOP<double>& D, thrust::complex<double> *M, Lattice& lattice, int const nBlocks_dot, int const nThreads_dot, int const nBlocks_drift, int const nThreads_drift){
	
	int const vol = lattice.vol;
	Spinor<double> *afterCG, *buf, *vec;
	thrust::complex<double> *eobuf;
	cudaMallocManaged(&afterCG, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&buf, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&vec, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&eobuf, sizeof(thrust::complex<double>) * 4 * vol);

	std::random_device rd; 
	std::mt19937 gen(rd()); 
	std::normal_distribution<float> dist(0, 1.0); 
	for(int i=0; i<vol; i++){ 
		afterCG[i].setZero(); 
		buf[i].setZero();
		for(int j=0; j<4; j++) vec[i].val[j] = dist(gen);
	}

	CGsolver_solve_D(vec, buf, D, M, nBlocks_dot, nThreads_dot);
	
	MatrixType useDagger = MatrixType::Dagger;
	void *diagArgs[] = {(void*)&buf, (void*)&afterCG, (void*) &vol, (void*) &D.fermion_mass, (void*) &g_coupling, (void*)&useDagger, (void*)&M};
	// hopping should be passed to all the off-diagonal (in spacetime) functions: Deo, Doe
	void *hoppingArgs[] = {(void*)&buf, (void*) &afterCG, (void*) &lattice.vol, (void*) &useDagger, (void*) &lattice.IUP, (void*) &lattice.IDN};
	D.applyD(diagArgs, hoppingArgs);
	cudaDeviceSynchronize();

	// Set up dot product call
	void *driftArgs[] = {(void*) &vec, (void*) &afterCG, (void*) &eobuf, (void*) &vol};
	auto dimGrid = dim3(nBlocks_drift, 1, 1);
	auto dimBlock = dim3(nThreads_drift, 1, 1);

	cudaLaunchCooperativeKernel((void*)&computeDrift, dimGrid, dimBlock, driftArgs, 0, NULL);
	cudaDeviceSynchronize();
	
	for(int i=0; i<4*vol; i++) outVec[i] = eobuf[NormalToEO(i, lattice)];

	cudaFree(afterCG);
	cudaFree(buf);
	cudaFree(vec);
	cudaFree(eobuf);
	 
}

__global__ void computeDrift(Spinor<double> *inVec, Spinor<double> *afterCG, thrust::complex<double> *outVec, int const vol){

	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();

	thrust::complex<double> im (0.0, 1.0);
	double const g_coupling = 0.1;

	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		// Drift for sigma
		outVec[i] = g_coupling * (	      conj(afterCG[i].val[0])*inVec[i].val[0]
										+ conj(afterCG[i].val[1])*inVec[i].val[1] 
										+ conj(afterCG[i].val[2])*inVec[i].val[2] 
										+ conj(afterCG[i].val[3])*inVec[i].val[3]);

		// Drift for pi1
		outVec[i + vol] = g_coupling * (	- conj(afterCG[i].val[0])*inVec[i].val[3]
											 	+ conj(afterCG[i].val[1])*inVec[i].val[2] 
												- conj(afterCG[i].val[2])*inVec[i].val[1] 
												+ conj(afterCG[i].val[3])*inVec[i].val[0]);

		// Drift for pi2
		outVec[i + 2*vol] = g_coupling * (	  im * conj(afterCG[i].val[0])*inVec[i].val[3] 
												- im * conj(afterCG[i].val[1])*inVec[i].val[2] 
												- im * conj(afterCG[i].val[2])*inVec[i].val[1] 
												+ im * conj(afterCG[i].val[3])*inVec[i].val[0]);

		// Drift for pi3
		outVec[i + 3*vol] = g_coupling * (	- conj(afterCG[i].val[0])*inVec[i].val[1]
												+ conj(afterCG[i].val[1])*inVec[i].val[0]
												+ conj(afterCG[i].val[2])*inVec[i].val[3]
												- conj(afterCG[i].val[3])*inVec[i].val[2]);

	}

}

__host__ __device__ int EOtoNormal(int n, Lattice& l){
	my2dArray idx = l.eoToVec(n);
	return idx[1] + l.Nx*idx[0];
}

__host__ __device__  int NormalToEO(int n, Lattice& l){
	int nt = n / l.Nx;
	int nx = n % Nx;
	return l.toEOflat(nt, nx);
}
