#include "Langevin.h"
#include <fstream>
#include <fftw3.h>


using namespace std;

Langevin::Langevin(O4Mesons* mesons) : 
	mesons{mesons},
    seed_gaussian(rd_gaussian()), 
    gaussian(0, 1), 
    seed_uniform(rd_uniform()),
    uniform(0, 1)
	{;}

Langevin::~Langevin(){};

void Langevin::LangevinRun(double dt, double T){

	int const Nt=mesons->Nt, Nx=mesons->Nx;
	
	for(double t=0; t<T; t+=dt){
		// Integrate Langevin equation with gaussian noise
		auto phicopy = mesons->phi;
		for(int nt=0; nt<Nt; nt++){
			for(int nx=0; nx<Nx; nx++){
				for(int j=0; j<4; j++){
					phicopy[nt][nx][j] = mesons->phi[nt][nx][j] + dt*mesons->evaluateDrift(nt, nx, j) + 0*sqrt(dt)*gaussian(seed_gaussian);
				}
			}
		}
		mesons->phi = phicopy;
	}
} 




// Prepare stuff for Fast Fourier Transform		
	/*fftw_complex phi_p[s->Nt][s->Nx], phi_x[s->Nt][s->Nx];	
	fftw_plan p,q;
	p = fftw_plan_dft_2d(s->Nt, s->Nx, phi_x[0], phi_p[0], FFTW_FORWARD, FFTW_MEASURE);
	q = fftw_plan_dft_2d(s->Nt, s->Nx, phi_p[0], phi_x[0], FFTW_BACKWARD, FFTW_MEASURE);*/

// Store field in the fftw variable (could be done via type cast)
	/*for(int nt=0; nt<s->Nt; nt++){
		for(int nx=0; nx<s->Nx; nx++){
			phi_x[nt][nx][0] = s->phi[nt][nx];
		}
	}	

	// transform
	fftw_execute(p);	
	
	// save phi(p) before cutoff
	if (t+dt>=T && n==Nvals-1){ for(int nt=0; nt<s->Nt; nt++){
		for(int nx=0; nx<s->Nx; nx++){
			if (nx != s->Nx-1) prep << ",";
		}
	}}
	// save phi(x) before cutoff 
	if (t+dt>=T && n==Nvals-1){ for(int nt=0; nt<s->Nt; nt++){
		for(int nx=0; nx<s->Nx; nx++){
			if (nx != s->Nx-1) prex << ",";
		}
	}}

	// Apply cutoff
	double s2p2 = cutoff_frac*cutoff_frac * 0.25 * (Nt*Nt + Nx*Nx);
	for(int nt=1; nt<=Nt/2; nt++){
		for(int nx=1; nx<=Nx/2; nx++){
			if (nx*nx + nt*nt >= s2p2) {
				phi_p[nt][nx][0] = 0.0; phi_p[nt][nx][1] = 0.0;
				phi_p[Nt-nt][nx][0] = 0.0; phi_p[Nt-nt][nx][1] = 0.0;
				phi_p[Nt-nt][Nx-nx][0] = 0.0; phi_p[Nt-nt][Nx-nx][1] = 0.0;
				phi_p[nt][Nx-nx][0] = 0.0; phi_p[nt][Nx-nx][1] = 0.0;
			}
		}
	}
	// do first row and column separately (momentum 0)
	for(int nt=1; nt<=Nt/2; nt++) {
		if (nt*nt >= s2p2) { 
			phi_p[nt][0][0]=0.0; phi_p[nt][0][1]=0.0; 
			phi_p[Nt-nt][0][0]=0.0; phi_p[Nt-nt][0][1]=0.0; 
		}
	}
	for(int nx=1; nx<=Nx/2; nx++) {
		if (nx*nx >= s2p2) { 
			phi_p[0][nx][0]=0.0; phi_p[0][nx][1]=0.0; 
			phi_p[0][Nx-nx][0]=0.0; phi_p[0][Nx-nx][1]=0.0; 
		}
	}
	
	// antitransform
	fftw_execute(q);
	
	// copy fftw variable into field
	for(int nt=0; nt<Nt; nt++){
		for(int nx=0; nx<Nx; nx++){
			s->phi[nt][nx] = (double) phi_x[nt][nx][0]/(s->Nt*s->Nx);
		}
	}	
	*/