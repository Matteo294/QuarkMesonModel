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
tr = np.average(data['tr'].to_numpy())


print("avg sigma:", sigma, np.std(data['sigma'].to_numpy()))
print("avg trace:", tr, np.std(data['tr'].to_numpy()))
print("sigma/tr --> measured:", sigma/tr, " expected: ", -g_coupling/mesons_mass)

