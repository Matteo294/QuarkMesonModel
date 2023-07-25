import numpy as np
from matplotlib import pyplot as plt 
from pandas import read_csv
import sys
import os
from matplotlib import rcParams

mypath = "s1/"

tr = []
tr_err = []
for direc in os.listdir(mypath):
    print(direc)
    os.chdir(mypath + direc)
    data = read_csv("traces.csv")
    N = len(data['tr'])
    tr.append(np.average(np.abs(data['phi'])))
    tr_err.append(np.std(np.abs(data['phi'])) / np.sqrt(N - 1))
    os.chdir("../../")

mphi = [10.0, 5.0, 2.0, 1.0, 0.5]

pairs = [(t, te) for t, te in zip(tr, tr_err)]
pairs = sorted(pairs, key = lambda x: x[0])

tr = [np.sqrt(m)*el[0]/np.sqrt(2) for m,el in zip(mphi,pairs)]
tr_err = [np.sqrt(m)*el[1]/np.sqrt(2) for m,el in zip(mphi,pairs)]

plt.errorbar(mphi, tr, tr_err, fmt='x--', label=r'$s = 1.0$')


mypath = "s2/"

tr = []
tr_err = []
for direc in os.listdir(mypath):
    print(direc)
    os.chdir(mypath + direc)
    data = read_csv("traces.csv")
    N = len(data['tr'])
    tr.append(np.average(np.abs(data['phi'])))
    tr_err.append(np.std(np.abs(data['phi'])) / np.sqrt(N - 1))
    os.chdir("../../")

mphi = [10.0, 5.0, 2.0, 1.0, 0.5]

pairs = [(t, te) for t, te in zip(tr, tr_err)]
pairs = sorted(pairs, key = lambda x: x[0])

tr = [np.sqrt(m)*el[0]/np.sqrt(2) for m,el in zip(mphi,pairs)]
tr_err = [np.sqrt(m)*el[1]/np.sqrt(2) for m,el in zip(mphi,pairs)]

plt.errorbar(mphi, tr, tr_err, fmt='x--', label=r'$s = 0.0$')


plt.ylabel(r'$\sqrt{\frac{m_\phi^2}{2}} \left\langle|\phi|\right\rangle$', fontsize=16)
plt.xlabel(r'$m_\phi^2$', fontsize=16)
plt.tight_layout()
plt.legend(fontsize=14)
plt.savefig("trace.pdf")
plt.show()