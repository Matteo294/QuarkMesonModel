import numpy as np
from matplotlib import pyplot as plt 
from pandas import read_csv
import sys
import os
from matplotlib import rcParams

rcParams['text.usetex'] = True

mypath = "./data/"

m0q = [100, 50, 20, 10, 5, 1]

phi = {}
phi_err = {}
for direc in os.listdir(mypath):
    os.chdir("data/" + direc)
    s = direc.split("_")[1]
    print("cutoff frac:", s)
    phi_s = []
    phi_s_err = []
    for config in sorted(os.listdir("./")):
        os.chdir(config)
        data = read_csv("traces.csv")
        N = len(data['phi'])
        print("Phi:", np.average(data['phi'].to_numpy()))
        phi_s.append(np.average(data['phi'].to_numpy()))
        phi_s_err.append(np.std(data['phi'].to_numpy()) / np.sqrt(N-1))
        os.chdir("..")
    phi[s] = phi_s
    phi_err[s] = phi_s_err
    os.chdir("..")
    os.chdir("..")
    print()
    

'''for s in phi.keys():
    plt.errorbar(sorted(m0q), phi[s], fmt='.-', yerr=phi_err[s], capsize=4, markersize=8, label=r'$s \, = \, ' + str(s) + '$')'''
    #plt.plot(m0q, phi[s], '-.', label='s=' + str(s))
plt.errorbar(m0q, phi["1.0"], fmt='.-', yerr=phi_err["1.0"], capsize=4, markersize=8, label=r'$s \, = \, ' + str(1.0) + '$')
plt.errorbar(m0q[:3], phi["0.5"], fmt='.-', yerr=phi_err["0.5"], capsize=4, markersize=8, label=r'$s \, = \, ' + str(0.5) + '$')
plt.xscale('log')
plt.legend()
plt.xlabel(r'$s \, m_q$')
plt.ylabel(r'$\left\langle|\phi|\right\rangle$')
plt.savefig('rescaling.pdf')
