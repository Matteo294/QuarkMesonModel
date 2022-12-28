from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit as fit
import numpy as np

#fit(lambda x, m: np.cosh(m*(x))

data = read_csv("data.csv") 

plt.subplot(2, 1, 1)
plt.plot(range(len(data['psi1'])), data['psi1'])
plt.xlabel('Nt')

plt.subplot(2, 1, 2)
plt.plot(range(len(data['psi2'])), data['psi2'])
plt.xlabel('Nt')

plt.suptitle("Spinor components of the correlator")
plt.show()