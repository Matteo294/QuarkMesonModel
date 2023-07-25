import numpy as np
from matplotlib import pyplot as plt 
from pandas import read_csv
import sys
import os
from matplotlib import rcParams
import toml

plot_diff = False
resc = True

diff_param = {}
phi_l = {}
phi_h = {}

# For s = 1.0
Nt = 8
Nx = 32

for direc in os.listdir("./data"):

    if (resc is True and float(direc.split('_')[1]) >= 1.0) or (resc is False and float(direc.split('_')[1]) <= 1.0):
        print(direc)
        os.chdir("data/" + direc)
        data = []
        for config in sorted(os.listdir("./")):
            os.chdir(config)
            params = toml.load("input.toml")
            s = float(params['physics']['cutFraction'])
            yukawa = float(params['fermions']['yukawa_coupling'])
            if yukawa == 0.0:
                yukawa = 1e-5
            csv_data = read_csv("traces.csv")
            phi = csv_data['phi'].to_numpy()
            print(s, yukawa, np.average(phi))
            N = len(phi)
            data.append( (yukawa, np.average(phi), np.std(phi) / np.sqrt(N)) )
            os.chdir("../")

        data = sorted(data, key =lambda x: x[0])

        param = [d[0] for d in data]
        val = [d[1] for d in data]
        err = [d[2] for d in data]
        lab = "s=" + str(s)
        if 'low' in direc:
            phi_l[str(s)] = val
            if str(s) not in diff_param.keys():
                diff_param[str(s)] = param
        elif 'high' in direc:
            phi_h[str(s)] = val
            if str(s) not in diff_param.keys():
                diff_param[str(s)] = param
        plt.errorbar(param, val, err, fmt='o--', label=lab, linewidth=2.0, markersize=8)

        os.chdir("../../")
        print()

plt.xlabel("g", fontsize=14)
plt.ylabel(r"$\left\langle |\phi| \right\rangle$", fontsize=14)
plt.xscale('log')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("phi.pdf")
plt.show()

if plot_diff:
    for s in phi_h.keys():
        phi = [h - l for h, l in zip(phi_h[str(s)], phi_l[str(s)])]
        plt.plot(diff_param[str(s)], phi, 'o--', label=str(s), markersize=8, linewidth=2.0)
    plt.legend()
    plt.xscale('log')
    plt.show()
        