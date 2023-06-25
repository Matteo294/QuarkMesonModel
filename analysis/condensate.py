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
'''mypath = "./mass/"
onlyfiles_mass = [f for f in listdir(mypath) if isfile(join(mypath, f))]'''

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
print(s)

'''m0 = []
for f in onlyfiles_mass:
    m = float(f.split('_')[1])
    m0.append(m)'''
    
    
mypath = "./data/"
for cutoff_frac in sorted(s):
    filename = mypath + "traces_" + str(cutoff_frac) + "_.csv"
    
    print("\n Cutoff fraction:", cutoff_frac)
   
    data = read_csv(filename)
    #data = data.drop(labels=range(1,1000), axis=0)
    N = len(data['sigma'])

    plt.hist(data['sigma'].to_numpy(), bins=50)
    plt.savefig("plots/sigma_hist/sigmhist_" + str(cutoff_frac) + ".pdf")
    plt.close()

    # vals
    sigma_vals.append(np.average(np.abs(data['sigma'].to_numpy())))
    pi_vals[0].append(np.average(np.abs(data['pi1'].to_numpy())))
    pi_vals[1].append(np.average(np.abs(data['pi2'].to_numpy())))
    pi_vals[2].append(np.average(np.abs(data['pi3'].to_numpy())))
    tr_vals.append(np.average(np.abs(data['tr'].to_numpy())))
    trpi_vals[0].append(np.average(np.abs(data['trp1'].to_numpy())))
    trpi_vals[1].append(np.average(np.abs(data['trp2'].to_numpy())))
    trpi_vals[2].append(np.average(np.abs(data['trp3'].to_numpy())))
    
    print("sigma:", np.average(data['sigma'].to_numpy()))
    print("pions:", np.average(data['pi1'].to_numpy()), np.average(data['pi2'].to_numpy()), np.average(data['pi3'].to_numpy()))
    

    # errs
    sigma_errs.append(np.std(np.abs(data['sigma'].to_numpy())) / np.sqrt(N-1))
    pi_errs[0].append(np.std(np.abs(data['pi1'].to_numpy())) / np.sqrt(N-1))
    pi_errs[1].append(np.std(np.abs(data['pi2'].to_numpy())) / np.sqrt(N-1))
    pi_errs[2].append(np.std(np.abs(data['pi3'].to_numpy())) / np.sqrt(N-1))
    tr_errs.append(np.std(np.abs(data['tr'].to_numpy())) / np.sqrt(N-1))
    trpi_errs[0].append(np.std(np.abs(data['trp1'].to_numpy())) / np.sqrt(N-1))
    trpi_errs[1].append(np.std(np.abs(data['trp2'].to_numpy())) / np.sqrt(N-1))
    trpi_errs[2].append(np.std(np.abs(data['trp3'].to_numpy())) / np.sqrt(N-1))

    plt.plot(range(len(sigma_vals)), sigma_vals, label=r'$\sigma$')
    plt.xlabel("Langevin time", fontsize=22)
    plt.title(r"$s=\Lambda' / \Lambda$ = " + str(cutoff_frac), fontsize=22)
    plt.tight_layout()
    plt.savefig("plots/pi_evol/sigmaevol_" + str(cutoff_frac) + ".pdf")
    plt.close()

    plt.plot(range(len(pi_vals[0])), pi_vals[0], label=r'$\pi_1$')
    plt.plot(range(len(pi_vals[1])), pi_vals[1], label=r'$\pi_2$')
    plt.plot(range(len(pi_vals[2])), pi_vals[2], label=r'$\pi_3$')
    plt.xlabel("Langevin time", fontsize=22)
    plt.title(r"$s=\Lambda' / \Lambda$ = " + str(cutoff_frac), fontsize=22)
    plt.tight_layout()
    plt.savefig("plots/sigma_evol/pievol_" + str(cutoff_frac) + ".pdf")
    plt.close()

    pisq = [x*x + y*y + z*z for x, y, z in zip(data['pi1'], data['pi2'], data['pi3'])]
    plt.plot(range(len(pisq)), pisq)
    plt.xlabel("Langevin time", fontsize=22)
    plt.ylabel(r"$|\pi|^2$", fontsize=18)
    plt.title(r"$s=\Lambda' / \Lambda$ = " + str(cutoff_frac), fontsize=22)
    plt.tight_layout()
    plt.savefig("plots/pi2_evol/pi2_evol_" + str(cutoff_frac) + ".pdf")
    plt.close()

print(len(s), len(sigma_vals))

# Condensates plot
plt.errorbar(sorted(s), np.abs(sigma_vals), fmt='.-', yerr=sigma_errs, capsize=4, markersize=4, label=r'$\left\langle|\sigma|\right\rangle$')
plt.errorbar(sorted(s), np.abs(tr_vals), fmt='.-', yerr=tr_errs, capsize=4, markersize=4, label=r'$\left\langle|\bar\psi\psi|\right\rangle$')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
plt.tight_layout()
#plt.ylim([0.78, 0.79])
plt.savefig("plots/condensates/sigma.pdf")
plt.close()

plt.errorbar(sorted(s), np.abs(pi_vals[0]), fmt='.-', yerr=pi_errs[0], capsize=4, markersize=12, label=r'$\left\langle|\pi_1|\right\rangle$')
plt.errorbar(sorted(s), np.abs(trpi_vals[0]), fmt='.-', yerr=trpi_errs[0], capsize=4, markersize=12, label=r'$\left\langle|\bar\psi\,\gamma_5\tau_1\,\psi|\right\rangle$')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
plt.tight_layout()
plt.yscale('log')
#plt.ylim([0.0, 0.02])
plt.savefig("plots/condensates/pi1.pdf")
plt.close()

plt.errorbar(sorted(s), np.abs(pi_vals[1]), fmt='.-', yerr=pi_errs[1], capsize=4, markersize=12, label=r'$\left\langle|\pi_2|\right\rangle$')
plt.errorbar(sorted(s), np.abs(trpi_vals[1]), fmt='.-', yerr=trpi_errs[1], capsize=4, markersize=12, label=r'$\left\langle|\bar\psi\,\gamma_5\tau_2\,\psi|\right\rangle$')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
plt.yscale('log')
plt.tight_layout()
#plt.ylim([0.577, 0.579])
plt.savefig("plots/condensates/pi2.pdf")
plt.close()

plt.errorbar(sorted(s), np.abs(pi_vals[2]), fmt='.-', yerr=pi_errs[2], capsize=4, markersize=12, label=r'$\left\langle|\pi_3|\right\rangle$')
plt.errorbar(sorted(s), np.abs(trpi_vals[2]), fmt='.-', yerr=trpi_errs[2], capsize=4, markersize=12, label=r'$\left\langle|\bar\psi\,\gamma_5\tau_3\,\psi\right\rangle$')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
plt.yscale('log')
plt.tight_layout()
#plt.ylim([0.28, 0.283])
plt.savefig("plots/condensates/pi3.pdf")
plt.close()


plt.errorbar(sorted(s), np.abs(pi_vals[0]), fmt='.-', yerr=pi_errs[0], capsize=4, markersize=12, label=r'$\left\langle|\pi_1|\right\rangle$')
plt.errorbar(sorted(s), np.abs(pi_vals[1]), fmt='.-', yerr=pi_errs[1], capsize=4, markersize=12, label=r'$\left\langle|\pi_2|\right\rangle$')
plt.errorbar(sorted(s), np.abs(pi_vals[2]), fmt='.-', yerr=pi_errs[2], capsize=4, markersize=12, label=r'$\left\langle|\pi_3|\right\rangle$')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
#plt.yscale('log')
plt.tight_layout()
#plt.ylim([0, 0.03])
plt.savefig("plots/pions.pdf")
plt.close()

sigma_vals = []
sigma_errs = []
pi_vals = [[] for _ in range(3)]
pi_errs = [[] for _ in range(3)]

'''mypath = "./mass/"
for m in sorted(m0):
    filename = mypath + "traces_" + str(m) + "_.csv"
   
    data = read_csv(filename)
    N = len(data['sigma'])

    # vals
    sigma_vals.append(np.average(data['sigma'].to_numpy()))
    pi_vals[0].append(np.average(data['pi1'].to_numpy()))
    pi_vals[1].append(np.average(data['pi2'].to_numpy()))
    pi_vals[2].append(np.average(data['pi3'].to_numpy()))

    # errs
    sigma_errs.append(np.std(data['sigma'].to_numpy()) / np.sqrt(N-1))
    pi_errs[0].append(np.std(data['pi1'].to_numpy()) / np.sqrt(N-1))
    pi_errs[1].append(np.std(data['pi2'].to_numpy()) / np.sqrt(N-1))
    pi_errs[2].append(np.std(data['pi3'].to_numpy()) / np.sqrt(N-1))

 
 
plt.errorbar(sorted(m0), np.abs(sigma_vals), fmt='.-', yerr=sigma_errs, capsize=4, markersize=12, label=r'$|\left\langle\sigma\right\rangle|$')
plt.grid()
plt.xscale('log')
plt.legend(fontsize=18)
plt.xlabel(r"$m_0$", fontsize=22)
plt.tight_layout()
plt.savefig("massplots/sigma.pdf")
plt.xscale('log')
plt.close()
   
plt.errorbar(sorted(m0), np.abs(pi_vals[0]), fmt='.-', yerr=pi_errs[0], capsize=4, markersize=12, label=r'$|\left\langle\pi_1\right\rangle|$')
plt.errorbar(sorted(m0), np.abs(pi_vals[1]), fmt='.-', yerr=pi_errs[1], capsize=4, markersize=12, label=r'$|\left\langle\pi_2\right\rangle|$')
plt.errorbar(sorted(m0), np.abs(pi_vals[2]), fmt='.-', yerr=pi_errs[2], capsize=4, markersize=12, label=r'$|\left\langle\pi_3\right\rangle|$')
plt.xscale('log')
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$m_0$", fontsize=22)
plt.tight_layout()
plt.savefig("massplots/pions.pdf")
plt.close()'''

