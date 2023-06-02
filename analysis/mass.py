from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit as fit
import numpy as np
import sys

'''m0 = 0.2
yukawa_coupling = 1.0
sigma = 0.2
pi = [0.3, 0.4, 0.1]'''

m0 = 0.2
yukawa_coupling = 1.0
sigma = 0.32
pi = [0.23, 0.11, 0.12]

plot = True

data = read_csv(sys.argv[1])
Nt = int(sys.argv[2])

def expectedM(m0, g, sigma, pi):
	r2 = sigma**2 + pi[0]**2 + pi[1]**2 + pi[2]**2
	denom = 2*(g*sigma + m0 + 1)
	sqrroot = np.sqrt((g**2*r2 + 2*m0*(g*sigma + 1) + 2*g*sigma + m0**2 + 2)**2 - 4*(g*sigma+m0+1)**2)
	num = -sqrroot + g**2*r2 + 2*g*m0*sigma + 2*g*sigma + m0**2 + 2*m0 + 2
	return -np.log(num/denom)

def fitToSinh(ydata, startidx, endidx, plot=False):
	yvals = ydata[startidx:endidx]
	xvals = np.array(range(startidx, endidx))

	def fitfuncSinh(x, m_re, A):
		return A * np.sinh(m_re*(Nt/2-x))

	fitparams = fit(fitfuncSinh, xvals, yvals, p0=[np.log(1+0.2), 1.0], maxfev=int(1e5))
	#print("Mass: ", abs(fitparams[0][0]))

	if plot:
		xvals = np.linspace(0, 128, 1000)
		plt.plot(xvals, fitfuncSinh(xvals, fitparams[0][0], fitparams[0][1]), label='fit')
		plt.plot(range(1,Nt), timeslices[-1][1:], label='data')
		plt.legend()
		plt.tight_layout()
		#plt.savefig("mass.pdf")
		plt.show()

	return abs(fitparams[0][0])

	'''plt.plot(xvals, fitfuncSinh(xvals, fitparams[0][0], fitparams[0][1]), label="fit")
	plt.plot(xvals, yvals, '.', markersize=6, label="data")
	plt.legend()
	plt.title("Temporal correlator m0=" + str(m0) + " g=" + str(g) + r" $\phi=[$" + str(sigma) + ", " + str(pi[0]) + ", " + str(pi[1]) + ", " + str(pi[2]) + "]")
	plt.xlabel("t")
	plt.show()'''
    
timeslices = data['f0c0'].to_numpy().reshape((-1, Nt))
print(timeslices.shape)

plt.plot(range(1,Nt), timeslices[-1][1:])
plt.title("f=0 c=0")
plt.xlabel('t')
plt.title("vecfield components of the correlator")
plt.tight_layout()
plt.show()

corr = np.average(timeslices, axis=0)
print(corr.shape)


finalmass = fitToSinh(corr, 2, Nt-2, plot)
print("mass:", finalmass)
print("Expected: ", expectedM(m0, yukawa_coupling, sigma, pi))





