#if !defined(USE_cuFFT) && !defined(USE_vkFFT)
#define USE_cUFFT
#endif

// Remember to add modify to drift instead of assign and remove magnetization

#include <cmath>
#include <atomic>
#include <vector>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <cooperative_groups.h>

//#include "HDF5.h"
#include "toml.hpp"
#include "params.h"
#include "Laplace.h"
#include "device_info.h"
#include "managedVector.h"
#include "colouredNoise.h"
#include "reductions.cuh"
#include "langevin_gpu_v2.cuh"

// ----------------------------------------------------------
#include "Dirac.cuh"
#include "Spinor.cuh"
#include "Lattice.cuh"
#include "CGsolver.cuh"
#include "FermionicDrift.cuh"
// ----------------------------------------------------------


__constant__ myType epsBar;
__constant__ myType m2;
__constant__ myType lambda;



// ----------------------------------------------------------
__constant__ double yukawa_coupling_gpu;
__constant__ double fermion_mass_gpu;
__constant__ thrust::complex<double> im_gpu;

__global__ void copyScalarsIntoM(double* phi, thrust::complex<double>* M);
// ----------------------------------------------------------




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
    srand(1234);
    
    std::cout << "Nt: " << Sizes[0] << " Nx: " << Sizes[1] << "\n";  

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
	auto const useMass = toml::find<std::string>(parameters, "useMass");
	myType my_m2, myLambda, kappa, Lambda;
	if (useMass == "true") {
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

	//auto hdf = HDF{outFileName, resumeRun};
	//hdf.Close();

	/*if (resumeRun == true) {
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
	}*/
	//

	// ----------------------------------------------------------
    int const spinor_vol = 4 * vol;

	auto const& fermionsSection = toml::find(inputData, "fermions");
	double const fermion_mass = toml::find<double>(fermionsSection, "fermion_mass");
	double const yukawa_coupling = toml::find<double>(fermionsSection, "yukawa_coupling");
    
	Spinor<double> *in, *out, *temp1, *temp2, *temp3;
	DiracOP<double> Dirac;
	FermionicDrift fDrift(seed);
	thrust::complex<double> *M;
	double *fermionic_contribution;
	CGsolver CG;
    double *traces; // trace D^-1
	int myvol; // dynamic volume

	cudaMallocManaged(&M, sizeof(thrust::complex<double>) * 4 * vol);
	cudaMallocManaged(&fermionic_contribution, sizeof(double) * 4 * vol);
	cudaMallocManaged(&in, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&out, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&traces, sizeof(double) * 4);
	cudaMallocManaged(&temp1, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&temp2, sizeof(Spinor<double>) * vol);
	cudaMallocManaged(&temp3, sizeof(Spinor<double>) * vol);


	Dirac.setM(M);
    for(int i=0; i<vol; i++) {
        for(int j=0; j<4; j++) {
            in[i].val[j] = 0.0;
            out[i].val[j] = 0.0;
        }
    }
    
    int nBlocks = 0;
	int nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, setZeroGPU);
	cudaDeviceSynchronize();
	auto dimGrid_zero = dim3(nBlocks, 1, 1);
	auto dimBlock_zero = dim3(nThreads, 1, 1);
    void *setZeroArgs[] = {(void*) &in, (void*) &myvol};
    
    nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, copyScalarsIntoM);
	cudaDeviceSynchronize();
	auto dimGrid_mesons = dim3(nBlocks, 1, 1);
	auto dimBlock_mesons = dim3(nThreads, 1, 1);
    void *copyMesonsArgs[] = {(void*) &in, (void*) &out};

	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, computeDrift);
	cudaDeviceSynchronize();
	auto dimGrid_drift = dim3(nBlocks, 1, 1);
	auto dimBlock_drift = dim3(nThreads, 1, 1);

	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, copyVec_double);
	cudaDeviceSynchronize();
	auto dimGrid_copyDouble = dim3(nBlocks, 1, 1);
	auto dimBlock_copyDouble = dim3(nThreads, 1, 1);
    void *copyVecDoubleArgs[] = {(void*) &in, (void*) &out, (void*) &spinor_vol};

	nBlocks = 0;
	nThreads = 0;
	cudaOccupancyMaxPotentialBlockSize(&nBlocks, &nThreads, gpuTraces);
	cudaDeviceSynchronize();
	auto dimGrid_traces = dim3(nBlocks, 1, 1);
	auto dimBlock_traces = dim3(nThreads, 1, 1);
    void *tracesArgs[] = {(void*) &fermionic_contribution, (void*) &traces};

	// set up print files
	std::ofstream datafile, tracefile;
	datafile.open("data.csv");
	datafile << "corr" << "\n";
	std::string fname;
	fname.append("traces"); fname.append(".csv");
	tracefile.open(fname);
	tracefile << "tr,trp1,trp2,trp3,sigma,pi1,pi2,pi3,phi" << "\n";
	// ----------------------------------------------------------

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
	// -----------------------------------------------------------------
	cudaMemcpyToSymbol(yukawa_coupling_gpu, &yukawa_coupling, sizeof(double));
	cudaMemcpyToSymbol(fermion_mass_gpu, &fermion_mass, sizeof(double));
	cudaMemcpyToSymbol(im_gpu, &im, sizeof(thrust::complex<double>));
	// -----------------------------------------------------------------


	// burn in a little bit, since the drift might be stronger at the beginning, since we are
	// likely far from the equilibrium state
	for (int burn = 0; burn < burnCount; ++burn) {
		myType t = 0.0;
		while (t < ExportTime) {
			cn();

			// ----------------------------------------------------------
			copyMesonsArgs[0] = (void*) &(ivec.data());
        	copyMesonsArgs[1] = (void*) &M;
			cudaLaunchCooperativeKernel((void*)copyScalarsIntoM, dimGrid_mesons, dimBlock_mesons, copyMesonsArgs, 0, NULL);
			cudaDeviceSynchronize();
			fDrift.getForce(fermionic_contribution, Dirac, M, CG, dimGrid_drift, dimBlock_drift);
            //for(int i=0; i<4*vol; i++) fermionic_contribution[i] = 0.0;
			copyVecDoubleArgs[0] = (void*) &(drift.data());
			copyVecDoubleArgs[1] = (void*) &fermionic_contribution;
			cudaLaunchCooperativeKernel((void*) copyVec_double, dimGrid_copyDouble, dimBlock_copyDouble, copyVecDoubleArgs, 0, NULL);
			cudaDeviceSynchronize();
            auto err = cudaPeekAtLastError();
            if (err != 0) std::cout << "CUDA ERROR: " << err << std::endl;
			// ----------------------------------------------------------

			kli.Run(kAll, kli_sMem);
			cudaDeviceSynchronize();
			cudaMemcpy(h_eps, eps, sizeof(myType), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
			t += *h_eps;
		}
	}
	
	std::cout << "Thermalization done!" << std::endl;

	int nMeasurements = 0;
	int oldMeasurements = 0;
	elapsedLangevinTime = 0.0;
	/*if (resumeRun == true) {
		hdf.Open();
		auto configName = hdf.NameOfLastConfig();
//		std::cout << "name of last config " << configName << '\n';
		hdf.ReadData("/data/", configName + "/fields", ivec);
		hdf.ReadSeeds("/seeds/", "last", cn.GetState());
		configName.erase(0, configName.find('_') + 1);
		oldMeasurements = std::stoi(configName);
		hdf.Close();
	}*/

	if (MeasureDriftCount > 0) {
	myType epsSum = 0.0;
	while (elapsedLangevinTime < MeasureDriftCount * ExportTime) {
		myType t = 0.0;
		while (t < ExportTime) {
			cn();

			// ----------------------------------------------------------
			copyMesonsArgs[0] = (void*) &(ivec.data());
        	copyMesonsArgs[1] = (void*) &M;
			cudaLaunchCooperativeKernel((void*) copyScalarsIntoM, dimGrid_mesons, dimBlock_mesons, copyMesonsArgs, 0, NULL);
			cudaDeviceSynchronize();
			fDrift.getForce(fermionic_contribution, Dirac, M, CG, dimGrid_drift, dimBlock_drift);
			copyVecDoubleArgs[0] = (void*) &(drift.data());
			copyVecDoubleArgs[1] = (void*) &fermionic_contribution;
			cudaLaunchCooperativeKernel((void*) copyVec_double, dimGrid_copyDouble, dimBlock_copyDouble, copyVecDoubleArgs, 0, NULL);
			cudaDeviceSynchronize();
			// ----------------------------------------------------------

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
    
    std::cout << "Drift measuring done!" << std::endl;

	// main loop
	elapsedLangevinTime = 0.0;
	nMeasurements = oldMeasurements;
	std::vector<myType> hostLattice(N*nVectorComponents);
	elapsedLangevinTime = nMeasurements * ExportTime;
	auto timeSliceFile = std::ofstream(timeSliceFileName);
	auto timerStart = std::chrono::high_resolution_clock::now();
    
    //elapsedLangevinTime = MaxLangevinTime;
    
	while (elapsedLangevinTime < MaxLangevinTime) {
		myType t = 0.0;
		while (t < ExportTime) {
			cn();

			// ----------------------------------------------------------
			copyMesonsArgs[0] = (void*) &(ivec.data());
        	copyMesonsArgs[1] = (void*) &M;
			cudaLaunchCooperativeKernel((void*) copyScalarsIntoM, dimGrid_mesons, dimBlock_mesons, copyMesonsArgs, 0, NULL);
			cudaDeviceSynchronize();
			fDrift.getForce(fermionic_contribution, Dirac, M, CG, dimGrid_drift, dimBlock_drift);
			copyVecDoubleArgs[0] = (void*) &(drift.data());
			copyVecDoubleArgs[1] = (void*) &fermionic_contribution;
			cudaLaunchCooperativeKernel((void*) copyVec_double, dimGrid_copyDouble, dimBlock_copyDouble, copyVecDoubleArgs, 0, NULL);
			cudaDeviceSynchronize();
			// ----------------------------------------------------------

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
			if (useMass == "false") e /= sq2Kappa;
			std::cout << e / N << '\t';
			sum2 += e*e;
		}
		std::cout << std::sqrt(sum2) / N << std::endl;
        
        // ------------------------------------------------------
        /*setZeroArgs[0] = (void*) &in;
        cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);        
		cudaDeviceSynchronize();

        in[0].val[0] = 1.0;
		in[0].val[1] = 1.0;
		in[0].val[2] = 1.0;
		in[0].val[3] = 1.0;
        
        copyMesonsArgs[0] = (void*) &(ivec.data());
        copyMesonsArgs[1] = (void*) &M;
        cudaLaunchCooperativeKernel((void*)&copyScalarsIntoM, dimGrid_mesons, dimBlock_mesons, copyMesonsArgs, 0, NULL);
        cudaDeviceSynchronize();

        switch (CGmode) {
			
			case '0':

				CG.solve(in, out, Dirac, MatrixType::Normal);

				myvol = spinor_vol;
				setZeroArgs[0] = (void*) &in;
				cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);
				cudaDeviceSynchronize();

				Dirac.setInVec(out);
				Dirac.setOutVec(in);
				Dirac.setDagger(MatrixType::Dagger);
				Dirac.applyD();
				cudaDeviceSynchronize();

				break;

			case '1':

				myvol = spinor_vol/2;
				
				//Dirac_d.D_oo_inv(psiField.pos.begin() + lattice.vol/2, temp2.begin());
				//Dirac_d.D_eo(temp2.begin(), temp1.begin());
				Dirac.setInVec(in); Dirac.setOutVec(temp2);
				cudaLaunchCooperativeKernel((void*)&D_oo_inv<double>, Dirac.dimGrid_Doo_inv, Dirac.dimBlock_Doo_inv, Dirac.diagArgs, 0, NULL);
				cudaDeviceSynchronize();
				Dirac.setInVec(temp2); Dirac.setOutVec(temp1);
				cudaLaunchCooperativeKernel((void*)&D_eo<double>, Dirac.dimGrid_Deo, Dirac.dimBlock_Deo, Dirac.diagArgs, 0, NULL);
				cudaDeviceSynchronize();

				//spinorDiff(psiField.pos.begin(), psiField.pos.begin() + lattice.vol/2, temp1.begin(), temp2.begin());
				for(int i=0; i<vol/2; i++) {
					for(int j=0; j<4; j++) temp2[i].val[j] = in[i].val[j] - temp1[i].val[j];
				}
				
				//CGdouble.solve_Dhat(temp2.begin(), temp2.end(), temp3.begin());
				CG.solveEO(temp2, temp3, Dirac, MatrixType::Normal);

				//for(int i=0; i<lattice.vol/2; i++) std::fill(psiField.pos[i].val.begin(), psiField.pos[i].val.begin(), 0.0);
				setZeroArgs[0] = (void*) &in;
				cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);        
				cudaDeviceSynchronize();

				//Dirac_d.applyDhatTo(temp3.begin(), psiField.pos.begin(), MatrixType::Dagger);
				Dirac.setDagger(MatrixType::Dagger);
				Dirac.applyDhat(temp3, in);

				//std::fill(temp1.begin(), temp1.end(), Spinor_d());
				setZeroArgs[0] = (void*) &temp1;
				cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);        
				cudaDeviceSynchronize();

				//Dirac_d.D_oe(psiField.pos.begin(), temp1.begin());
				Dirac.setInVec(in); Dirac.setOutVec(temp1);
				cudaLaunchCooperativeKernel((void*)&D_oe<double>, Dirac.dimGrid_Doe, Dirac.dimBlock_Doe, Dirac.hoppingArgs, 0, NULL);
				cudaDeviceSynchronize();

				//spinorDiff(psiField.pos.begin() + lattice.vol/2, psiField.pos.end(), temp1.begin(), temp3.begin());
				for(int i=0; i<vol/2; i++){
					for(int j=0; j<4; j++) temp3[i].val[j] = in[i+vol/2].val[j] - temp1[i].val[j];
				}

				//for(int i=lattice.vol/2; i<lattice.vol; i++) std::fill(psiField.pos[i].val.begin(), psiField.pos[i].val.end(), 0.0);
				setZeroArgs[0] = (void*) &in;
				setZeroArgs[1] = (void*) &myvol;
				cudaLaunchCooperativeKernel((void*)&setZeroGPU, dimGrid_zero, dimBlock_zero, setZeroArgs, 0, NULL);        
				cudaDeviceSynchronize();


				//Dirac_d.D_oo_inv(temp3.begin(), psiField.pos.begin() + lattice.vol/2);
				Dirac.setInVec(temp3); Dirac.setOutVec(temp1);
				cudaLaunchCooperativeKernel((void*)&D_oo_inv<double>, Dirac.dimGrid_Doo_inv, Dirac.dimBlock_Doo_inv, Dirac.diagArgs, 0, NULL);
				cudaDeviceSynchronize();

				break;

		}

		thrust::complex<double> corr = 0.0;
		for(int nt=0; nt<Sizes[0]; nt++){
			corr = 0.0;
			for(int nx=0; nx<Sizes[1]; nx++){
				for(int j=0; j<4; j++) corr += in[toEOflat(nt, nx)].val[j];
			}
			datafile << corr.real() << "\n";
		}*/
        
        // compute condensates from drifts as they are proportional
        for(int i=0; i<4; i++) traces[i] = 0.0;
        
        //for(int i=0; i<4*vol; i++) traces[(int) i/vol] += fermionic_contribution[i];
		cudaLaunchCooperativeKernel((void*) &gpuTraces, dimGrid_traces, dimBlock_traces, tracesArgs, 32 * sizeof(double), NULL);
		cudaDeviceSynchronize();
        
        if (yukawa_coupling != 0.0) { 
            for(int i=0; i<4; i++) traces[i] /= yukawa_coupling;
        } else {
            for(int i=0; i<4; i++) traces[i] = 0.0;
        }
		
		for(int i=0; i<4; i++) tracefile << (double) traces[i] / vol << ",";
        for(int i=0; i<4; i++) tracefile << (double) avg[i] / vol << ",";
		tracefile << std::sqrt(sum2) / vol << "\n";
		// ------------------------------------------------------

		nMeasurements++;
		
		// this explicit copy seems to peform slightly/marginally better
		// TODO: needs further investigation
		cudaMemcpy(hostLattice.data(), ivec.data(), N*nVectorComponents*sizeof(myType),
				cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		// if the user provided kappa as input, we rescale the output field to dimensionless format
		if (useMass == "false")
			for (auto& e : hostLattice)
				// divide or multiply...?
				e /= sq2Kappa;
		if (early_finish == true) {
			std::cout << "#Early termination signal received.\n#Wrapping up.\n";
			elapsedLangevinTime = MaxLangevinTime + 1.0;
		}
		/*std::stringstream ss;
		ss << "data/cnfg_" << std::setfill('0') << std::setw(8) << 
			(exportHDF == true ? nMeasurements : 1);
		hdf.Open();
		hdf.CreateGroup(ss.str());
		hdf.WriteData(ss.str(), "fields", hostLattice);
		hdf.Close();*/
	}
	auto timerStop  = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart);
	timeSliceFile.close();
	/*hdf.Open();
	hdf.WriteSeeds("/seeds", "last", cn.GetState());
	hdf.Close();*/

	std::cout << "#numSms = " << kli.numSms << '\n';
	std::cout << "#blocks per SM = " << kli.numBlocksPerSm << '\n';
	std::cout << "#theads = " << kli.numThreads << '\n';
	std::cout << "#blocks = " << kli.numBlocks << '\n';

	std::cout << "#Number of measurements: " << nMeasurements << '\n';

	std::cout << "#Run time for main loop: " << duration.count() / 1000.0 << "s\n";

	cudaFree(eps);
	cudaFree(maxDrift);
	free(h_eps);

	// ------------------------------------------------
	cudaFree(M);
	cudaFree(in);
	cudaFree(out);
	cudaFree(fermionic_contribution);
	cudaFree(traces);
	cudaFree(temp1);
	cudaFree(temp2);
	cudaFree(temp3);
	// ------------------------------------------------
    
    
	return 0;
}

__global__ void copyScalarsIntoM(double* phi, thrust::complex<double>* M){
    int eo_i;
    auto grid = cg::this_grid();
    thrust::complex<double> const im {0.0, 1.0};
	for (int i = grid.thread_rank(); i < vol; i += grid.size()){
        eo_i = NormalToEO(i);
        M[eo_i] = phi[i] + im * phi[3*vol+i];
        M[eo_i + 3*vol] = phi[i] - im * phi[3*vol+i];
        M[eo_i + vol] = im * (phi[vol+i] - im * phi[2*vol+i]);
        M[eo_i + 2*vol] = im * (phi[vol+i] + im * phi[2*vol+i]);
    }
}
    
