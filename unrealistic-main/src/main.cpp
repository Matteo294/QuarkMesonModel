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

__constant__ myType epsBar;
__constant__ myType m2;
__constant__ myType lambda;

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
		hdf.CreateGroup("/seeds");
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
	auto timerStart = std::chrono::high_resolution_clock::now();
	while (elapsedLangevinTime < MaxLangevinTime) {
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

//		cudaMemPrefetchAsync(ivec.data(), N*nVectorComponents, cudaCpuDeviceId);
		cudaLaunchCooperativeKernel((void*)gpuMagnetisation, kli.dimGrid, kli.dimBlock,
				kMagnetisation, kli_sMem, NULL);
		cudaDeviceSynchronize();

		kTimeSlices.Run(timeSlicesArgs, kli_sMem);
		cudaDeviceSynchronize();

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

	return 0;
}
