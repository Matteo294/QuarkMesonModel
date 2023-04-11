import numpy as np
from matplotlib import pyplot as plt 

# Mass plot classical
mq0 = 	 [5.0,		2.0, 		1.0,		0.5,		0.2]
mq = 	 [1.7917594,	1.08306,	0.6639, 	0.35921,	0.16518]		
mq_dev = [1e-17,	1e-5,		0.0001,    	1e-5,		1e-5]

plt.errorbar([np.log(1+m) for m in mq0], mq, yerr=mq_dev, fmt='.-', markersize=12, capsize=4)
plt.xlabel("mq0")
plt.ylabel("mq")
plt.show()

# Mass analysis
mq0 = [1.0, 0.8, 0.6, 0.4]
sigma = [0.1308, 0.1391, 0.1483, 0.1574]
pi1 = [0.0669, 0.0712, 0.0763, 0.0807]
pi2 = [0.0006, 0.0010, 0.0013, 0.0014]
pi3 = [0.0002, 0.0003, 1.3e-5, 0.0002]

plt.subplot(2,1,1)
plt.plot(mq0, sigma, '.-')
plt.xlabel("mq0")
plt.ylabel("sigma")

plt.subplot(2,1,2)
plt.plot(mq0, pi1, '.-', label='pi1')
plt.plot(mq0, pi2, '.-', label='pi2')
plt.plot(mq0, pi3, '.-', label='pi3')
plt.xlabel("mq0")
plt.legend()
plt.show()



plt.show()

del sigma
del pi1
del pi2
del pi3


# Condensate plot

s = 		[1.0,    0.8, 	 0.6, 	 0.4, 	 0.2, 	 0.0]
sigma_vals = 	[0.129,  0.129,  0.129,  0.129,  0.129,  0.130116]
sigma_err = 	[0.001,  0.001,  0.001,  0.001,  0.001,  8e-6]
trace_vals = 	[0.1303, 0.1302, 0.1303, 0.1304, 0.1302, 0.1301]
trace_err = 	[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
ratios = [sv/tv for sv,tv in zip(sigma_vals, trace_vals)]
ratios_err = [0.001 for i in range(6)]

plt.errorbar(s, ratios, fmt='.-', yerr=ratios_err, capsize=4, markersize=12)
plt.plot([0.0, 1.0], [1.0, 1.0], '--', linewidth=1.5, color='black', alpha=0.7)
plt.grid()
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
plt.ylabel(r'$- \ \frac{\left\langle\sigma\right\rangle}{\left\langle\bar\psi \psi\right\rangle}$', fontsize=22, rotation=0, labelpad=40)
plt.tight_layout()
plt.show()

plt.errorbar(s, sigma_vals, fmt='.-', markersize=12, yerr=sigma_err, capsize=4, label=r'-$<\sigma>$', color='blue')
#plt.plot(s, sigma_vals, '-', linewidth=1.5, color='blue')
plt.errorbar(s, trace_vals, fmt='.-', markersize=12, yerr=trace_err, capsize=4, label=r'$<trace>$', color='orange')
#plt.plot(s, trace_vals, '-', linewidth=1.5, color='orange')
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=18)
plt.grid()
plt.ylim([0.127, 0.132])
plt.tight_layout()
plt.show()
