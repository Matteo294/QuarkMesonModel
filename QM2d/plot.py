from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit as fit
import numpy as np

# Read configuration settings
params = {}
with open("conffile.txt") as paramsfile:
    for line in paramsfile.readlines():
        x = line.split("=")
        val = str(x[1]).replace("\n", "")
        params[x[0]] = float(val)
    

m0 = params['m0']
g = params['g']
sigma = params['sigma']
pi = [params['pi1'], params['pi2'], params['pi3']]
print("\nConfiguration: ")
for p in params:
    print(p + ": ", params[p])
print()

data = read_csv("data.csv") 

Nt = len(data['f0c0'])

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

    fitparams = fit(fitfuncSinh, xvals, yvals, p0=[np.log(1+m0), 1.0])
    print("Mass: ", abs(fitparams[0][0]))
    print("Expected: ", expectedM(m0, g, sigma, pi))

    plt.plot(xvals, fitfuncSinh(xvals, fitparams[0][0], fitparams[0][1]), label="fit")
    plt.plot(xvals, yvals, '.', markersize=6, label="data")
    plt.legend()
    plt.title("Temporal correlator m0=" + str(m0) + " g=" + str(g) + r" $\phi=[$" + str(sigma) + ", " + str(pi[0]) + ", " + str(pi[1]) + ", " + str(pi[2]) + "]")
    plt.xlabel("t")
    plt.show()

def fitToExp(ydata, startidx, endidx):
    yvals = ydata[startidx:endidx]
    xvals = np.array(range(startidx, endidx))

    def fitfuncExp(x, m_re, A):
        return A * np.exp(-m_re*x)

    fitparams = fit(fitfuncExp, xvals, yvals)
    print("Mass: ", fitparams[0][0])
    print("Expected: ", expectedM(m0, g, sigma, pi))

    plt.plot(xvals, fitfuncExp(xvals, fitparams[0][0], fitparams[0][1]), label="fit")
    plt.plot(xvals, yvals, '.', markersize=12, label="data")
    plt.legend()
    plt.show()

#plt.subplot(2, 2, 1)
plt.plot(range(Nt), data['f0c0'])
plt.title("f=0 c=0")
plt.xlabel('t')

'''plt.subplot(2, 2, 2)
plt.plot(range(Nt), data['f0c1'])
plt.title("f=0 c=1")
plt.xlabel('t')

plt.subplot(2, 2, 3)
plt.plot(range(Nt), data['f1c0'])
plt.title("f=1 c=0")
plt.xlabel('t')

plt.subplot(2, 2, 4)
plt.plot(range(Nt), data['f1c1'])
plt.title("f=1 c=1")
plt.xlabel('t')'''

#plt.suptitle("vecfield components of the correlator")
plt.title("vecfield components of the correlator")
plt.tight_layout()
plt.show()


fitToSinh(data['f0c0'], 1, Nt)
#fitToSinh(data['f0c1'], 1, Nt)
#fitToSinh(data['f1c0'], 1, Nt)
#fitToSinh(data['f1c1'], 1, Nt)

#fitToExp(data['f0c0'], 1, Nt)
#fitToExp(data['f0c1'], 4, Nt)
#fitToExp(data['f1c0'], 8, Nt)
#fitToExp(data['f1c1'], 16, Nt)

