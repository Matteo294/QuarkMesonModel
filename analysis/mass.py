from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit as fit
import numpy as np
import sys


data = read_csv(sys.argv[1])
Nt = int(sys.argv[2])

def expectedM(m0, g, sigma, pi):
    r2 = sigma**2 + pi[0]**2 + pi[1]**2 + pi[2]**2
    denom = 2*(g*sigma + m0 + 1)
    sqrroot = np.sqrt((g**2*r2 + 2*m0*(g*sigma + 1) + 2*g*sigma + m0**2 + 2)**2 - 4*(g*sigma+m0+1)**2)
    num = -sqrroot + g**2*r2 + 2*g*m0*sigma + 2*g*sigma + m0**2 + 2*m0 + 2
    return -np.log(num/denom)

def fitToSinh(ydata, startidx, endidx):
    yvals = ydata[startidx:endidx]
    xvals = np.array(range(startidx, endidx))

    def fitfuncSinh(x, m_re, A):
        return A * np.sinh(m_re*(Nt/2-x))

    fitparams = fit(fitfuncSinh, xvals, yvals, p0=[np.log(1+2.0), 1.0], maxfev=int(1e5))
    #print("Mass: ", abs(fitparams[0][0]))
    #print("Expected: ", expectedM(0.1, 1.0, -0.15927687234042556, [0, 0, 0]))
    
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


masses = []
for sl in timeslices:
	masses.append(fitToSinh(sl, 1, Nt))
	
finalmass = np.average(masses)
print("mass:", finalmass, "+-", np.std(masses)/np.sqrt(len(masses)))


