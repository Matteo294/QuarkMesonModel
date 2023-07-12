import numpy as np
from matplotlib import pyplot as plt 
from pandas import read_csv
import sys
import os
from matplotlib import rcParams

rcParams['text.usetex'] = True

mypath = "./data/"

phi_h = {}
phi_err_h = {}
phi_l = {}
phi_err_l = {}
for direc in os.listdir(mypath):
    os.chdir("data/" + direc)
    s = direc.split("_")[1]
    #print("cutoff frac:", s)
    phi_s = {}
    phi_s_err = {}
    for config in sorted(os.listdir("./")):
        g = config.split("_")[1]
        os.chdir(config)
        data = read_csv("traces.csv")
        N = len(data['phi'])
        #print("Phi:", np.average(data['phi'].to_numpy()))
        phi_s[g] = np.average(data['phi'].to_numpy())
        phi_s_err[g] = np.std(data['phi'].to_numpy()) / np.sqrt(N-1)
        os.chdir("..")
    if direc.split("_")[2] == 'high':
        phi_h[s] = phi_s
        phi_err_h[s] = phi_s_err
    elif direc.split("_")[2] == 'low':
        phi_l[s] = phi_s
        phi_err_l[s] = phi_s_err
    else:
        print("Folder name not recognised")
    os.chdir("..")
    os.chdir("..")
    print()
   

for s in phi_h.keys():
	print("Cutoff", s)
	print("high")
	print(phi_h[s])
	print("low")
	print(phi_l[s])


for s in phi_h.keys():
    data = []
    for g in phi_h[s].keys():
        #data.append((float(g), phi_h[s][g] - phi_l[s][g], np.sqrt(phi_err_h[s][g]**2 + phi_err_l[s][g]**2)))
        #data.append((float(g), phi_h[s][g], np.sqrt(phi_err_h[s][g]**2)))
        data.append((float(g), phi_l[s][g], np.sqrt(phi_err_l[s][g]**2)))
    data = sorted(data, key = lambda x: x[0])
    
    yukawa = []
    phi = []
    err = []
    for tup in data:
        yukawa.append(tup[0])
        phi.append(tup[1])
        err.append(tup[2])
    plt.errorbar(yukawa, phi, fmt='x--', yerr=err, label='s=' + str(s))
    #phi_err[s] = [np.sqrt(h*h + l*l) for h,l in zip(phi_err_h[s], phi_err_l[s])]
#
# plt.errorbar(m0q, phi["1.0"], fmt='.-', yerr=phi_err["1.0"], capsize=4, markersize=8, label=r'$s \, = \, ' + str(1.0) + '$')
#plt.errorbar(m0q[:3], phi["0.5"], fmt='.-', yerr=phi_err["0.5"], capsize=4, markersize=8, label=r'$s \, = \, ' + str(0.5) + '$')
#plt.xscale('log')
plt.legend()
plt.xlabel(r'$g / s$')
plt.ylabel(r'$\left\langle|\phi|\right\rangle$')
plt.savefig('rescaling.pdf')
