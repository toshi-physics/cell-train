import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
plt.style.use("./tplot.mplstyle")

ta = 20
tc = 50
ab1 = 10.0
ab2 = 5.0
ab3 = 5.0
n=1000
qs = np.linspace(0,0.6,n)
roots = np.zeros((3,n),dtype='complex128')

for i in np.arange(len(qs)):
    disp = Polynomial([(ab3+1)*qs[i]*qs[i], 1+ qs[i]*qs[i]*(ta+tc), ta+tc+(qs[i]*qs[i]*ta*tc), ta*tc])
    roots[:,i] += disp.roots()
plt.axhline(y=0, color='k')
plt.plot(qs,roots[1].imag,c='#5a3173')
plt.plot(qs,roots[2].imag,c='#5a3173')
plt.plot(qs, roots[0].real,c='#ee5a31')
plt.plot(qs,roots[1].real,c='#ee5a31')
plt.plot(qs,roots[2].real,c='#ee5a31')
plt.ylabel(r'$\omega\tau$')
plt.xlabel(r'$q$')
plt.ylim(-0.15, 0.15)
plt.xlim(0,0.6)
#plt.show()
plt.savefig('dispersion_relation_5.svg')