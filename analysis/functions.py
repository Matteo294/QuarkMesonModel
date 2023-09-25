from pandas import read_csv
import toml
from scipy.optimize import curve_fit as fit
import numpy as np
import ctypes

lib_path = "/home/matteo/Downloads/ac/build/"

class dataset:
    
    # mode can be either 0 (no rescaling of params) or 1 (rescaling params / block spin)
    def add_data(self, folder, param, mode):
        fields_data = read_csv(folder + '/traces.csv')
        mass_data = read_csv(folder + '/data.csv')
        S_t = get_time_slices_from_timeslicefile(folder + "/slice.dat", field_axis=0, return_all=False)
        data_Sq_t = read_csv(folder + "/data.csv")
        Sq_t = data_Sq_t['corr'].to_numpy(np.dtype('f8')).reshape((-1, Nt))
        toml_params = get_toml_params(folder + '/input.toml')
        
        # magnetisation
        blob = ComputeStatistics(data['sigma'])
        self.phi.append((blob.average, blob.sigmaF))
        avgxxx = blob.average
        # condensate
        blob = ComputeStatistics(data['tr'])
        self.cond.append((blob.average, blob.sigmaF))
        # susceptibility
        blob = ComputeStatistics((data['sigma'] - avgxxx)**2)
        self.chi2.append((blob.average, blob.sigmaF))
        # renormalised bosonic mass
        val, err = get_ren_mass_right_via_timeslices(S_t,volume)
        self.m_phi_r.append((val, err))
        # physical quark mass
        val, err = get_phys_quark_mass_via_timeslices(Sq_t,volume)
        self.m_q_phys.append((val, err))
        # parameter value
        p = toml_params[param[0]][param[1]]
        s = toml_params['physics']['cutFraction']
        if mode == 1:
            if param[1] == "mass":
                p /= s*s
        self.parameters.append(p)
    
    def sort_data(self, data):
        arr = [(p, val) for p, val in zip(self.parameters, data)]
        sorted_arr = sorted(arr, key = lambda x: x[0])
        sorted_p = []
        sorted_val = []
        for x in sorted_arr:
            sorted_p.append(x[0])
            sorted_val.append(x[1])
        return sorted_p, sorted_val
    
    def get_toml_params(filename):
        params = toml.load("data/" + f + "/input.toml")
        return params
        
    def __init__(self, name):
        self.phi = [] # <phi>
        self.abs_phi = [] # <|phi|>
        self.chi2 = [] # susceptiblity or second connected moment
        self.m_phi_r = [] # renormalised mesons mass
        self.m_q_phys = [] # physical quark mass
        self.parameters = [] # x parameter
    


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

def fitToSinh(ydata, startidx, endidx, plot=False):
    yvals = ydata[startidx:endidx]
    xvals = np.array(range(startidx, endidx))

    fitparams = fit(fitfuncSinh, xvals, yvals)
        
    return fitparams[0]

def get_phys_quark_mass_via_timeslices(Sq_t, volume):
    global Nt 
    Nt = int(volume[0])
    corr = np.average(Sq_t, axis=0)
    try:
    	val, err = fitToSinh(corr, 1, Nt, plot=False)
    except:
    	val = 0
    	err = 0
    del Nt
    return [val, err]
