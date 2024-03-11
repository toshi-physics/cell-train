import numpy as np
from matplotlib import pyplot as plt
import os
plt.style.use("./tplot.mplstyle")

L=20
perturbation = 'steady_state'

ablist = np.array(os.listdir(path='{:s}/{:d}'.format(perturbation, L)))

t2 = np.array([1, 2, 20])
tr = np.array([1/20, 1/10, 2/5, 1, 5/2, 10, 20]) #t2/t3

abc = (np.sqrt(tr)+(1/np.sqrt(tr)))*(np.sqrt(tr)+(1/np.sqrt(tr))+2) #critical ab

b = np.array([0.1, 1, 10]) #beta nondimensionalised
ab = np.array([1, 8, 9, 10, 11, 15]) #alpha nondim

abtr1 = np.array([[0],[0]])
abtr2 = np.array([[0],[0]])

for t2i in t2:
    for t3i in t2i/tr:
        for bi in b:
            abci = (np.sqrt(t2i/t3i)+(1/np.sqrt(t2i/t3i)))*(np.sqrt(t2i/t3i)+(1/np.sqrt(t2i/t3i))+2)
            for abi in np.arange(abci-4, abci+5, 2):
                if abi<abci:
                    abtr1 = np.append(abtr1, [[t2i/t3i],[abi]], axis=1)
                else:
                    abtr2 = np.append(abtr2,[[t2i/t3i],[abi]], axis=1)

trp = np.arange(0.02, 50, 0.001)
fig, ax = plt.subplots(1)
ax.plot(trp,  (np.sqrt(trp)+(1/np.sqrt(trp)))*(np.sqrt(trp)+(1/np.sqrt(trp))+2), color='black', linewidth=1.5)
ax.scatter(abtr1[0, 1:], abtr1[1, 1:], c='#5a3173',marker='x')
ax.scatter(abtr2[0, 1:], abtr2[1, 1:], c='#ee5a31', marker='o')
ax.set_xscale('log')
ax.tick_params(which='both', right=True, left=True, top=True, bottom=True, direction='in')
ax.set_ylim(0,40)
ax.set_xlabel(r'$\tau_a/\tau_c$')
ax.set_ylabel(r'$\alpha\beta$')
plt.savefig('phase_plot.svg')