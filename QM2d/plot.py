from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit as fit
import numpy as np

Mf = 0.5 + 1



data = read_csv("data.csv") 

plt.subplot(2, 1, 1)
plt.plot(range(len(data['psi1'])), data['psi1'])
plt.xlabel('Nt')

plt.subplot(2, 1, 2)
plt.plot(range(len(data['psi2'])), data['psi2'])
plt.xlabel('Nt')

plt.suptitle("Spinor components of the correlator")
plt.show()

Nt = len(data["psi1"]) 
yvals = data['psi1'][1:] # do not include first point in the fit
fitfunc = lambda x, m, A: A * np.sinh(m*(Nt/2-x))
xvals = np.array(range(1, Nt))


fitparams = fit(fitfunc, xvals, yvals)

print("Mass: ", fitparams[0][0])
print("Expected: ", np.log(1+Mf))

plt.plot(xvals, fitfunc(xvals, fitparams[0][0], fitparams[0][1]))
plt.plot(xvals, yvals)
plt.show()