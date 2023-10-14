from pandas import read_csv
import toml
from scipy.optimize import curve_fit as fit
import numpy as np
import ctypes

from correlations import *
from read_in_data import *

lib_path = "/home/matteo/code/QuarkMesonModel/analysis/AutoCorrelation/build/"

class Dataset:
    
    # mode can be either 0 (no rescaling of params) or 1 (rescaling params / block spin)
    def add_data(self, folder, param, mode):
        self.fields_data = read_csv(folder + '/traces.csv')
        try:
            self.fields_data = read_csv(folder + '/traces.csv')
        except:
            print("Reading traces.csv was not possible")
        try:
            self.S_t = get_time_slices_from_timeslicefile(folder + "/slice.dat", field_axis=0, return_all=False)
        except:
            print("Reading slice.dat was not possible")
        try:
            self.data_Sq_t = read_csv(folder + "/data.csv")
        except:
            print("Reading data.csv was not possible")
        try:
            self.Sq_t = self.data_Sq_t['corr'].to_numpy(np.dtype('f8')).reshape((-1, self.Nt))
        except:
            print("Reshaping fermionic correlator was not possible")
             
        self.toml_params = self.get_toml_params(folder + '/input.toml')
        self.volume = (self.Nt, self.Nx)
        self.mode = mode
    
    def compute_mag(self, printing=False):
        blob = ComputeStatistics(self.fields_data['sigma'])
        if printing:
            print("Npoints:", len(self.fields_data['sigma']), " \t val:", blob.average, "+-", blob.sigmaF, "\t ACtime:", blob.ACtime)
        return blob.average, blob.sigmaF
    
    def compute_abs_mag(self, printing=False):
        blob = ComputeStatistics(self.fields_data['phi'])
        if printing:
            print("Npoints:", len(self.fields_data['sigma']), " \t val:", blob.average, "+-", blob.sigmaF, "\t ACtime:", blob.ACtime)
        return blob.average, blob.sigmaF
    
    def compute_condensate(self, printing=False):
        blob = ComputeStatistics(self.fields_data['tr'])
        if printing:
            print("Npoints:", len(self.fields_data['sigma']), " \t val:", blob.average, "+-", blob.sigmaF, "\t ACtime:", blob.ACtime)
        return blob.average, blob.sigmaF
            
    def compute_susceptibility(self, printing=False):
        blob = ComputeStatistics(self.fields_data['sigma'])
        blob = ComputeStatistics((self.fields_data['sigma'] - blob.average)**2)
        if printing:
            print("Npoints:", len(self.fields_data['sigma']), " \t val:", blob.average, "+-", blob.sigmaF, "\t ACtime:", blob.ACtime)
        return blob.average, blob.sigmaF
    
    def compute_mphir(self, printing=False):
        val, err = get_ren_mass_right_via_timeslices(self.S_t, self.volume)
        if printing:
            print("Npoints:", len(self.fields_data['sigma']), " \t val:", val, "+-", err)
        return val, err
    
    def compute_mqphys(self, printing=False):
        val, err = get_fermionic_correlator(self.Sq_t)
        val, err = get_phys_quark_mass_via_sinh(val, self.volume, plotting=False)
        if printing:
            print(p[1], "Npoints:", len(self.fields_data['sigma']), " \t val:", val, "+-", err)
        return val, err
    
    def get_toml_params(self, filename):
        params = toml.load(filename)
        return params
        
    def __init__(self, Nt, Nx):
        self.Nt = Nt
        self.Nx = Nx
        self.volume = (self.Nt, self.Nx)

def sort_data(data):
        sorted_arr = sorted(data, key = lambda x: x[0])
        sorted_p = []
        sorted_val = []
        sorted_err = []
        for x in sorted_arr:
            sorted_p.append(x[0])
            sorted_val.append(x[1])
            sorted_err.append(x[2])
        return sorted_p, sorted_val, sorted_err
    


class ACreturn(ctypes.Structure):
	_fields_ = [ ("average", ctypes.c_double), ("sigmaF", ctypes.c_double), ("sigmaF_error", ctypes.c_double), ("ACtime", ctypes.c_double), ("ACtimeError", ctypes.c_double) ]


def ComputeStatistics(data):
	ft_lib = ctypes.cdll.LoadLibrary(lib_path + "libac.so")

	arg = (ctypes.c_double * len(data))(*data)
	avg = ft_lib.Wrapper
	avg.restype = ACreturn

	return avg(arg, len(data))

def expectedM(m0, g, sigma, pi):
    r2 = sigma**2 + pi[0]**2 + pi[1]**2 + pi[2]**2
    denom = 2*(g*sigma + m0 + 1)
    sqrroot = np.sqrt((g**2*r2 + 2*m0*(g*sigma + 1) + 2*g*sigma + m0**2 + 2)**2 - 4*(g*sigma+m0+1)**2)
    num = -sqrroot + g**2*r2 + 2*g*m0*sigma + 2*g*sigma + m0**2 + 2*m0 + 2
    return -np.log(num/denom)

def fitfuncSinh(x, m_re, A):
    return A * np.sinh(m_re*(Nt/2-x))

def fitfuncExp(x, m_re, A):
    return A * np.exp(-x*m_re)

def fitToExp(ydata, startidx, endidx, plotting):
    yvals = ydata[startidx:endidx]
    xvals = np.array(range(startidx, endidx))

    fitparams = fit(fitfuncExp, xvals, yvals, p0=[1.0, 1.0], maxfev=5000)
    
    if plotting is True:
        plt.plot(xvals, fitfuncExp(xvals, fitparams[0][0], fitparams[0][1]), label="fit")
        plt.plot(xvals, yvals, '.', markersize=6, label="data")
        plt.legend()
        plt.xlabel("t")
        plt.show()
        
    return fitparams[0]

def fitToSinh(ydata, startidx, endidx, plotting):
    yvals = ydata[startidx:endidx]
    xvals = np.array(range(startidx, endidx))

    fitparams = fit(fitfuncSinh, xvals, yvals, p0=[1.0, 1.0], maxfev=5000)
    
    if plotting is True:
        plt.plot(xvals, fitfuncSinh(xvals, fitparams[0][0], fitparams[0][1]), label="fit")
        plt.plot(xvals, yvals, '.', markersize=6, label="data")
        plt.legend()
        plt.xlabel("t")
        plt.show()
        
    return fitparams[0]

def get_fermionic_correlator(Sq_t):
    val = np.average(Sq_t, axis=0)
    err = np.std(Sq_t, axis=0)
    return val, err
    
def get_phys_quark_mass_via_sinh(corr, volume,  startidx=1, endidx=None, plotting=True, printing=True):
    if endidx is None:
        endidx = volume[0] - 1
    global Nt
    Nt = int(volume[0])
    val, err = fitToSinh(corr, startidx, endidx, plotting=plotting)
    if printing:
        print(val, err)
    del Nt
    return val, err

def get_phys_quark_mass_via_exp(corr, volume, startidx=0, endidx=None, plotting=True, printing=True):
    if endidx is None:
        endidx = volume[0] - 1
    global Nt
    Nt = int(volume[0])
    val, err = fitToExp(corr, startidx, endidx, plotting=plotting)
    if printing:
        print(val, err)
    del Nt
    return val, err


def get_ren_mass_right_via_timeslices2(S_t, volume, chi2, chi_err):

    Nt = volume[0]
    Nx = volume[1]
    
    a = np.arange(0, Nt)
    a[a>int(Nt/2)] = Nt-a[a>int(Nt/2)]


    connected_corr_t, connected_corr_t_err = get_connected_2pt_fct(S_t)
    #chi2 = Nx * np.sum( connected_corr_t)
    mu2 = 2 * Nx * np.sum( connected_corr_t * a**2)
    
    ren_mass2 = 2*2 * chi2/mu2
    ren_mass = np.sqrt( ren_mass2 )
        
    #chi_err = Nx * np.sqrt( np.sum( connected_corr_t_err**2 ))
    mu_err = 2*Nx * np.sqrt( np.sum( (a**2 * connected_corr_t_err)**2  )   )
    mass2_err = ren_mass2 * np.sqrt( (chi_err/chi2)**2 + (mu_err/mu2)**2 )
    mass_err = ren_mass *0.5 *mass2_err/ren_mass2

    return ren_mass, mass_err
