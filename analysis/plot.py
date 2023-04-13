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
mq0 = [10.0, 5.0, 2.0, 1.0, 0.8, 0.6, 0.4, 0.2]
sigma = [0.0325, 0.0563, 0.0991, 0.1308, 0.1391, 0.1483, 0.1574, 0.1630]
pi1 = [0.0176, 0.0296, 0.0669, 0.0712, 0.0763, 0.0807, 0.0841, 0.0843]
pi2 = [0.0002, 0.0003, 0.0006, 0.0010, 0.0013, 0.0014, 0.0026, 0.0043]
pi3 = [0.00056, 0.00063, 0.0002, 0.0003, 1.3e-5, 0.0002, 0.0002, 0.0001]

plt.subplot(2,2,1)
plt.plot(mq0, sigma, '.-')
plt.xlabel("mq0")
plt.ylabel("sigma")

plt.subplot(2,2,2)
plt.plot(mq0, pi1, '.-', label='pi1')
plt.legend()

plt.subplot(2,2,3)
plt.plot(mq0, pi2, '.-', label='pi2')
plt.legend()

plt.subplot(2,2,4)
plt.plot(mq0, pi3, '.-', label='pi3')
plt.legend()

plt.show()

del sigma


# Condensate plot

s = 		    [1.0,    0.8, 	 0.6, 	 0.4, 	 0.2, 	 0.0]

sigma_vals = 	[1.610,  1.614,  1.612,  1.613,  1.608,  1.624]
sigma_err = 	[0.003,  0.003,  0.003,  0.003,  0.003,  0.002]
trace_vals = 	[1.622, 1.625, 1.623, 1.627, 1.623, 1.625]
trace_err = 	[0.002, 0.002, 0.002, 0.002, 0.002, 0.002]

pi1_vals = [0.890, 0.891, 0.893, 0.892, 0.891, 0.886]
pi1_err = [0.003 for i in range(5)]
pi1_err.append(0.002)
trp1_vals = [0.885, 0.890, 0.886, 0.890, 0.889, 0.890]
trp1_err = [0.002 for i in range(5)]
trp1_err.append(0.0002)

pi2_vals = [0.179, 0.180, 0.177, 0.177, 0.177, 0.172]
pi2_err = [0.003 for i in range(5)]
pi2_err.append(0.0001)
trp2_vals = [0.175, 0.175, 0.176, 0.177, 0.174, 0.173]
trp2_err = [0.001 for i in range(5)]
trp2_err.append(0.0004)

pi3_vals = [0.018, 0.018, 0.018, 0.016, 0.017, 0.024]
pi3_err = [0.003 for i in range(5)]
pi3_err.append(1e-5)
trp3_vals = [0.025, 0.026, 0.027, 0.027, 0.025, 0.025]
trp3_err = [0.001 for i in range(5)]
trp3_err.append(0.0001)



plt.errorbar(s, sigma_vals, fmt='.-', yerr=sigma_err, capsize=4, markersize=12, label=r'$\left\langle\sigma\right\rangle$')
plt.errorbar(s, trace_vals, fmt='.-', yerr=trace_err, capsize=4, markersize=12, label=r'$\left\langle\bar\psi\psi\right\rangle$')
#plt.plot([0.0, 1.0], [1.0, 1.0], '--', linewidth=1.5, color='black', alpha=0.7)
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
#plt.ylabel(r'$- \ \frac{\left\langle\sigma\right\rangle}{\left\langle\bar\psi \psi\right\rangle}$', fontsize=22, rotation=0, labelpad=40)
plt.tight_layout()
plt.savefig("niceplots/sigma.pdf")
plt.show()

plt.errorbar(s, pi1_vals, fmt='.-', yerr=pi1_err, capsize=4, markersize=12, label=r'$\left\langle\pi_1\right\rangle$')
plt.errorbar(s, trp1_vals, fmt='.-', yerr=trp1_err, capsize=4, markersize=12, label=r'$\left\langle\bar\psi\,\gamma_5\tau_1\,\psi\right\rangle$')
#plt.plot([0.0, 1.0], [1.0, 1.0], '--', linewidth=1.5, color='black', alpha=0.7)
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
#plt.ylabel(r'$- \ \frac{\left\langle\sigma\right\rangle}{\left\langle\bar\psi \psi\right\rangle}$', fontsize=22, rotation=0, labelpad=40)
plt.tight_layout()
plt.savefig("niceplots/pi1.pdf")
plt.show()

plt.errorbar(s, pi2_vals, fmt='.-', yerr=pi2_err, capsize=4, markersize=12, label=r'$\left\langle\pi_2\right\rangle$')
plt.errorbar(s, trp2_vals, fmt='.-', yerr=trp2_err, capsize=4, markersize=12, label=r'$\left\langle\bar\psi\,\gamma_5\tau_2\,\psi\right\rangle$')
#plt.plot([0.0, 1.0], [1.0, 1.0], '--', linewidth=1.5, color='black', alpha=0.7)
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
#plt.ylabel(r'$- \ \frac{\left\langle\sigma\right\rangle}{\left\langle\bar\psi \psi\right\rangle}$', fontsize=22, rotation=0, labelpad=40)
plt.tight_layout()
plt.savefig("niceplots/pi2.pdf")
plt.show()

plt.errorbar(s, pi3_vals, fmt='.-', yerr=pi3_err, capsize=4, markersize=12, label=r'$\left\langle\pi_3\right\rangle$')
plt.errorbar(s, trp3_vals, fmt='.-', yerr=trp3_err, capsize=4, markersize=12, label=r'$\left\langle\bar\psi\,\gamma_5\tau_3\,\psi\right\rangle$')
#plt.plot([0.0, 1.0], [1.0, 1.0], '--', linewidth=1.5, color='black', alpha=0.7)
plt.grid()
plt.legend(fontsize=18)
plt.xlabel(r"$s=\frac{\Lambda'}{\Lambda}$", fontsize=22)
#plt.ylabel(r'$- \ \frac{\left\langle\sigma\right\rangle}{\left\langle\bar\psi \psi\right\rangle}$', fontsize=22, rotation=0, labelpad=40)
plt.tight_layout()
plt.savefig("niceplots/pi3.pdf")
plt.show()
