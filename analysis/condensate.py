import numpy as np
from matplotlib import pyplot as plt 
from pandas import read_csv
import sys

from matplotlib import rcParams
rcParams['text.usetex'] = True


data = read_csv(sys.argv[1])

mesons_mass = 1.0
g_coupling = 1.0

sigma = np.average(data['sigma'].to_numpy())
pi1 = np.average(data['pi1'].to_numpy())
pi2 = np.average(data['pi2'].to_numpy())
pi3 = np.average(data['pi3'].to_numpy())
tr = np.average(data['tr'].to_numpy())
trp1 = np.average(data['trp1'].to_numpy())
trp2 = np.average(data['trp2'].to_numpy())
trp3 = np.average(data['trp3'].to_numpy())


print("avg sigma:", sigma, np.std(data['sigma'].to_numpy()) / np.sqrt(len(data['sigma'])))
print("avg trace:", tr, np.std(data['tr'].to_numpy()) / np.sqrt(len(data['sigma'])), "\n")

print("avg pi1:", pi1, np.std(data['pi1'].to_numpy()) / np.sqrt(len(data['sigma'])))
print("avg trace:", trp1, np.std(data['trp1'].to_numpy()) / np.sqrt(len(data['sigma'])), "\n")

print("avg pi2:", pi2, np.std(data['pi2'].to_numpy()) / np.sqrt(len(data['sigma'])))
print("avg trace:", trp2, np.std(data['trp2'].to_numpy()) / np.sqrt(len(data['sigma'])), "\n")

print("avg pi3:", pi3, np.std(data['pi3'].to_numpy()) / np.sqrt(len(data['sigma'])))
print("avg trace:", trp3, np.std(data['trp3'].to_numpy()) / np.sqrt(len(data['sigma'])), "\n")


#print("sigma/tr --> measured:", sigma/tr, " expected: ", -g_coupling/mesons_mass)

