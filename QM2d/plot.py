from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit as fit
import numpy as np

#fit(lambda x, m: np.cosh(m*(x))

data = read_csv("data.csv")


plt.plot(range(len(data)), data['data'])
plt.show()