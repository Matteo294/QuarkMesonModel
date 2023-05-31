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
#include "CGsolver.cuh"
#include "FermionicDrift.cuh"

__constant__ myType epsBar;
__constant__ myType m2;
__constant__ myType lambda;
__constant__ double yukawa_coupling_gpu;
__constant__ double fermion_mass_gpu;

using std::conj;

__global__ void copyMesonsToM(double* phi, thrust::complex<double>* M);

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

	cudaMemcpyToSymbol(yukawa_coupling_gpu, &yukawa_coupling, sizeof(double));
	cudaMemcpyToSymbol(fermion_mass_gpu, &fermion_mass, sizeof(double));
	
	thrust::complex<double> *M;
	double *fermionic_contribution;
	Spinor<double> *in, *out;
	DiracOP<double> Dirac;
	FermionicDrift fDrift;
	CGsolver CG;

	

	// Allocate two vectors and mesons matrix
	cudaMallocManaged(&M, sizeof(thrust::complex<double>) * 4 * vol);
	cudaMallocManaged(&in, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&out, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&fermionic_contribution, sizeof(double) * 4 * vol);

	Dirac.setM(M);

	// Set up kernel function calls (NOTE!! the argument array is created here but modified immediately before function calls!)
	int nBlocks = 0;
	int nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, computeDrift);
	cudaDeviceSynchronize();
	auto dimGrid_drift = dim3(nBlocks, 1, 1);
	auto dimBlock_drift = dim3(nThreads, 1, 1);
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, setZeroGPU);
	cudaDeviceSynchronize();
	auto dimGrid_zero = dim3(nBlocks, 1, 1);
	auto dimBlock_zero = dim3(nThreads, 1, 1);
	int const spinor_vol = 4*vol;
	void *setZeroArgs[] = {(void*)out, (void*) &spinor_vol};
	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, copyVec);
	cudaDeviceSynchronize();
	auto dimGrid_copy = dim3(nBlocks, 1, 1);
	auto dimBlock_copy = dim3(nThreads, 1, 1);
	void *copyArgs[] = {(void*) &in, (void*) &out, (void*) &spinor_vol};
	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, copyMesonsToM);
	cudaDeviceSynchronize();
	auto dimGrid_copyM = dim3(nBlocks, 1, 1);
	auto dimBlock_copyM = dim3(nThreads, 1, 1);
	void *copyMArgs[] = {(void*) &in, (void*) &M};

	// set up print files
	std::ofstream datafile, tracefile;
	datafile.open("data.csv");
	datafile << "f0c0,f0c1,f1c0,f1c1" << "\n";
	std::string fname;
	fname.append("traces"); fname.append(".csv");
	tracefile.open(fname);
	tracefile << "tr,trp1,trp2,trp3,sigma,pi1,pi2,pi3" << "\n";
	// --------------------------------------------------------------
	
	for(int i=0; i<4*vol; i++) M[i] = 0.0;
	for(int i=0; i<vol; i++){in[i].setZero(); out[i].setZero();}

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
		(void*)&maxDrift};

	// ----------------------------------------------------------------------
	copyMArgs[0] = (void*)&(ivec.data());
	// ----------------------------------------------------------------------


	std::cout << std::endl;		// force a flush so we can see something on the screen before
								// actual computations start

	//

	cudaMemcpyToSymbol(m2, &my_m2, sizeof(myType));
	cudaMemcpyToSymbol(lambda, &myLambda, sizeof(myType));
	cudaMemcpyToSymbol(epsBar, &myEpsBar, sizeof(myType));
	cudaMemcpyToSymbol(yukawa_coupling_gpu, &yukawa_coupling, sizeof(double));
	cudaMemcpyToSymbol(fermion_mass_gpu, &fermion_mass, sizeof(double));

	// burn in a little bit, since the drift might be stronger at the beginning, since we are
	// likely far from the equilibrium state
	for (int burn = 0; burn < burnCount; ++burn) {
		myType t = 0.0;
		while (t < ExportTime) {
			cn();

			// ------------------------------------------------------------------------------------------------
			cudaLaunchCooperativeKernel((void*)&copyMesonsToM, dimGrid_copy, dimGrid_copy, copyMArgs, 0, NULL);
			cudaDeviceSynchronize();
			setZeroArgs[0] = (void*) &fermionic_contribution;
			cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
			cudaDeviceSynchronize();
			//fDrift.getForce(fermionic_contribution, Dirac, M, CG, dimGrid_drift, dimBlock_drift);
			/*copyArgs[0] = (void*) &drift;
			copyArgs[1] = (void*) &fermionic_contribution;
			cudaLaunchCooperativeKernel((void*) &copyVec_re, dimGrid_copy, dimBlock_copy, copyArgs, 0, NULL);
			cudaDeviceSynchronize();*/
			// for(int i=0; i<4*vol; i++){drift[i] = fermionic_contribution[i];}
			// ------------------------------------------------------------------------------------------------


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

			// ------------------------------------------------------------------------------------------------
			cudaLaunchCooperativeKernel((void*)&copyMesonsToM, dimGrid_copy, dimGrid_copy, copyMArgs, 0, NULL);
			cudaDeviceSynchronize();
			setZeroArgs[0] = (void*)&fermionic_contribution;
			cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
			cudaDeviceSynchronize();
			//fDrift.getForce(fermionic_contribution, Dirac, M, CG, dimGrid_drift, dimBlock_drift);
			/*copyArgs[0] = (void*) &drift;
			copyArgs[1] = (void*) &fermionic_contribution;
			cudaLaunchCooperativeKernel((void*)&copyVec_re, dimGrid_copy, dimBlock_copy, copyArgs, 0, NULL);
			cudaDeviceSynchronize();*/
			//for(int i=0; i<4*vol; i++){drift[i] = fermionic_contribution[i];}
			// ------------------------------------------------------------------------------------------------

			
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

	thrust::complex<double> tr[4]; // trace D^-1
	std::vector<thrust::complex<double>> traces;

	auto timerStart = std::chrono::high_resolution_clock::now();
	auto cycleStart = timerStart;
	auto cycleStop = timerStart;
	//while (elapsedLangevinTime < MaxLangevinTime) {
		myType t = 0.0;
		
		cycleStart = std::chrono::high_resolution_clock::now();
		while (t < ExportTime) {
			cn();			
			// ------------------------------------------------------------------------------------------------
			cudaLaunchCooperativeKernel((void*)&copyMesonsToM, dimGrid_copy, dimGrid_copy, copyMArgs, 0, NULL);
			cudaDeviceSynchronize();
			setZeroArgs[0] = (void*)&fermionic_contribution;
			cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
			cudaDeviceSynchronize();
			//fDrift.getForce(fermionic_contribution, Dirac, M, CG, dimGrid_drift, dimBlock_drift);
			/*copyArgs[0] = (void*) &drift;
			copyArgs[1] = (void*) &fermionic_contribution;
			cudaLaunchCooperativeKernel((void*)&copyVec_re, dimGrid_copy, dimBlock_copy, copyArgs, 0, NULL);
			cudaDeviceSynchronize();*/
			//for(int i=0; i<4*vol; i++){drift[i] = fermionic_contribution[i];}
			// ------------------------------------------------------------------------------------------------
			
			kli.Run(kAll, kli_sMem);
			cudaMemcpy(h_eps, eps, sizeof(myType), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t += *h_eps;
		//}
		cycleStop = std::chrono::high_resolution_clock::now();
		std::cout << "Cycle time: " << std::chrono::duration_cast<std::chrono::milliseconds>(cycleStop - cycleStart).count() / 1000.0 << " s \n";
		
		elapsedLangevinTime += t;

//		cudaMemPrefetchAsync(ivec.data(), N*nVectorComponents, cudaCpuDeviceId);
		cudaLaunchCooperativeKernel((void*)gpuMagnetisation, kli.dimGrid, kli.dimGrid,
				kMagnetisation, kli_sMem, NULL);
		cudaDeviceSynchronize();


		kTimeSlices.Run(timeSlicesArgs, kli_sMem);
		cudaDeviceSynchronize();


		// -----------------------------------------------------------------------------
		// Set fields values
		cudaLaunchCooperativeKernel((void*)&copyMesonsToM, dimGrid_copy, dimGrid_copy, copyMArgs, 0, NULL);
		cudaDeviceSynchronize();
		setZeroArgs[0] = (void*)&in;
		cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
		cudaDeviceSynchronize();
		setZeroArgs[0] = (void*)&out;
		cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
		cudaDeviceSynchronize();
		

		for(int i=0; i<vol; i++){
			M[i] = 0.2 + im * 0.1;
			M[i+3*vol] = 0.2 - im * 0.1;
			M[i+vol] = im * (0.1 - im * 0.15);
			M[i+2*vol] = im * (0.1 + im * 0.15);
			/*M[i] = 0.0;
			M[i+3*vol] = 0.0;
			M[i+vol] = 0.0;
			M[i+2*vol] = 0.0;*/
		}

		// Compute fermionic correlator and print to file
		in[0].val[0] = 1.0;
		in[0].val[1] = 1.0;
		in[0].val[2] = 1.0;
		in[0].val[3] = 1.0;

		CG.solve(in, out, Dirac);
		for(int i=0; i<vol; i++){in[i].setZero();}
		Dirac.setInVec(out);
		Dirac.setOutVec(in);
		Dirac.setDagger(MatrixType::Dagger);
		Dirac.applyD();

		thrust::complex<double> corr = 0.0;
		for(int nt=0; nt<Sizes[0]; nt++){
			corr = 0.0;
			for(int nx=0; nx<Sizes[1]; nx++){
				for(int j=0; j<4; j++) corr += in[toEOflat(nt, nx)].val[j];
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

		for(int i=0; i<4; i++) tr[i] = 0.0;
		for(int i=0; i<4*vol; i++) tr[(int) i/vol] += fermionic_contribution[i];
		tr[0] /= yukawa_coupling;
		tr[1] /= yukawa_coupling;
		tr[2] /= yukawa_coupling;
		tr[3] /= yukawa_coupling;
		tracefile 	<< 	tr[0].real()/(Sizes[0]*Sizes[1]) 	<< ","
					<< 	tr[1].real()/(Sizes[0]*Sizes[1]) 	<< ","
					<< 	tr[2].real()/(Sizes[0]*Sizes[1]) 	<< ","
					<< 	tr[3].real()/(Sizes[0]*Sizes[1]) 	<< ","
					<< 	avg[0] / (Sizes[0]*Sizes[1]) 		<< ","
					<< 	avg[1] / (Sizes[0]*Sizes[1]) 		<< "," 
					<< 	avg[2] / (Sizes[0]*Sizes[1]) 		<< ","
					<< 	avg[3] / (Sizes[0]*Sizes[1]) 		<< std::endl;
		std::cout << "Traces: " 	<< tr[0].real()/(Sizes[0]*Sizes[1]) << "\t" 
								<< tr[1].real()/(Sizes[0]*Sizes[1]) << "\t" 
								<< tr[2].real()/(Sizes[0]*Sizes[1]) << "\t"
								<< tr[3].real()/(Sizes[0]*Sizes[1]) << std::endl;

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
	cudaFree(fermionic_contribution);
	// ---------------------------------

	return 0;
}

__global__ void copyMesonsToM(double* phi, thrust::complex<double>* M){
	cg::thread_block cta = cg::this_thread_block();
	cg::grid_group grid = cg::this_grid();

	thrust::complex<double> const im {0.0, 1.0};


	int eo_i;
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
		eo_i = NormalToEO(i);
		M[eo_i] = phi[i] + im * phi[3*vol+i];
		M[eo_i + 3*vol] = phi[i] - im * phi[3*vol+i];
		M[eo_i + vol] = im * (phi[vol+i] - im * phi[2*vol+i]);
		M[eo_i + 2*vol] = im * (phi[vol+i] + im * phi[2*vol+i]);
	}	
	
}

