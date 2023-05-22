import numpy as np
from matplotlib import pyplot as plt 
from pandas import read_csv
import sys
from os import listdir
from os.path import isfile, join
from matplotlib import rcParams

rcParams['text.usetex'] = True

mypath = "./data/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

sigma_vals = []
sigma_errs = []
tr_vals = []
tr_errs = []

pi_vals = [[] for _ in range(3)]
pi_errs = [[] for _ in range(3)]
trpi_vals = [[] for _ in range(3)]
trpi_errs = [[] for _ in range(3)]

s = []
for f in onlyfiles:
    cutoff_frac = float(f.split('_')[1])
    s.append(cutoff_frac)

for cutoff_frac in sorted(s):
    filename = mypath + "traces_" + str(cutoff_frac) + "_.csv"
    data = read_csv(mypath + f)
    N = len(data['sigma'])

    '''plt.hist(data['sigma'].to_numpy(), bins=50)
    plt.show()'''

    # vals
    sigma_vals.append(np.average(data['sigma'].to_numpy()))
    pi_vals[0].append(np.average(data['pi1'].to_numpy()))
    pi_vals[1].append(np.average(data['pi2'].to_numpy()))
    pi_vals[2].append(np.average(data['pi3'].to_numpy()))
    tr_vals.append(np.average(data['tr'].to_numpy()))
    trpi_vals[0].append(np.average(data['trp1'].to_numpy()))
    trpi_vals[1].append(np.average(data['trp2'].to_numpy()))
    trpi_vals[2].append(np.average(data['trp3'].to_numpy()))

    # errs
    sigma_errs.append(np.std(data['sigma'].to_numpy()) / np.sqrt(N-1))
    pi_errs[0].append(np.std(data['pi1'].to_numpy()) / np.sqrt(N-1))
    pi_errs[1].append(np.std(data['pi2'].to_numpy()) / np.sqrt(N-1))
    pi_errs[2].append(np.std(data['pi3'].to_numpy()) / np.sqrt(N-1))
    tr_errs.append(np.std(data['tr'].to_numpy()) / np.sqrt(N-1))
    trpi_errs[0].append(np.std(data['trp1'].to_numpy()) / np.sqrt(N-1))
    trpi_errs[1].append(np.std(data['trp2'].to_numpy()) / np.sqrt(N-1))
    trpi_errs[2].append(np.std(data['trp3'].to_numpy()) / np.sqrt(N-1))

    pisq = [x*x + y*y + z*z for x, y, z in zip(data['pi1'], data['pi2'], data['pi3'])]
    plt.plot(range(len(pisq)), pisq)
    plt.xlabel("Langevin time", fontsize=22)
    plt.ylabel(r"$|\pi|^2$", fontsize=18)
    plt.title(r"$s=\Lambda' / \Lambda$ = " + str(cutoff_frac), fontsize=22)
    plt.tight_layout()
    plt.savefig("niceplots/pievol" + str(cutoff_frac) + ".pdf")
    plt.close()
    #plt.show()

print(len(s), len(sigma_vals))

# Condensates plot
plt.errorbar(s, np.abs(sigma_vals), fmt='.-', yerr=sigma_errs, capsize=4, markersize=12, label=r'$|\left\langle\sigma\right\rangle|$')
plt.errorbar(s, np.abs(tr_vals), fmt='.-', yerr=tr_errs, capsize=4, markersize=12, label=r'$|\left\langle\bar\psi\psi\right\rangle|$')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
plt.tight_layout()
#plt.ylim([1.252, 1.256])
plt.savefig("niceplots/sigma.pdf")
plt.show()

plt.errorbar(s, np.abs(pi_vals[0]), fmt='.-', yerr=pi_errs[0], capsize=4, markersize=12, label=r'$|\left\langle\pi_1\right\rangle|$')
plt.errorbar(s, np.abs(trpi_vals[0]), fmt='.-', yerr=trpi_errs[0], capsize=4, markersize=12, label=r'$|\left\langle\bar\psi\,\gamma_5\tau_1\,\psi\right\rangle|$')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
plt.tight_layout()
#plt.ylim([0.964, 0.966])
plt.savefig("niceplots/pi1.pdf")
plt.show()

plt.errorbar(s, np.abs(pi_vals[1]), fmt='.-', yerr=pi_errs[1], capsize=4, markersize=12, label=r'$|\left\langle\pi_2\right\rangle|$')
plt.errorbar(s, np.abs(trpi_vals[1]), fmt='.-', yerr=trpi_errs[1], capsize=4, markersize=12, label=r'$|\left\langle\bar\psi\,\gamma_5\tau_2\,\psi\right\rangle|$')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
plt.tight_layout()
#plt.ylim([0.577, 0.579])
plt.savefig("niceplots/pi2.pdf")
plt.show()

plt.errorbar(s, np.abs(pi_vals[2]), fmt='.-', yerr=pi_errs[2], capsize=4, markersize=12, label=r'$|\left\langle\pi_3\right\rangle|$')
plt.errorbar(s, np.abs(trpi_vals[2]), fmt='.-', yerr=trpi_errs[2], capsize=4, markersize=12, label=r'$|\left\langle\bar\psi\,\gamma_5\tau_3\,\psi\right\rangle|$')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
plt.tight_layout()
#plt.ylim([0.28, 0.283])
plt.savefig("niceplots/pi3.pdf")
plt.show()
