from cProfile import label
import os
import json
import numpy as np
from numba import jit
from def_integrator import integrator

if os.path.isfile('parameters_Model_B.json'):
	with open('parameters_Model_B.json') as jsonFile:
		parameters = json.load(jsonFile)

l = parameters["l"]
l0= parameters["l0"]/l #eqbm rest length nondim
t1 = parameters["t1"]
Dr0 = parameters["Dr0"] #displacement of first vertex attached to the bead
omega0 = t1*2*np.pi/parameters["T0"] #oscillation frequency of the external strain
tf = parameters["tf"]/t1
dt = parameters["timestep"]/t1
t = np.arange(0, tf+dt, dt)
# lh = l-(Dr0)/N # this makes lbar = L(t)/N
lh = l0 # this makes lbar L(0)/N


N = np.array([10, 15, 19, 20, 25, 40, 50, 75])  #number of cells (+1 is number of vertices), each cell is 20 microns, L=100, 200, 300, 380, 400, 500, 800, 1000, 1500, 2000
N = np.array([40, 50, 75])
t2 = np.array([1, 2, 20])/t1
tr = np.array([1/20, 1/10, 2/5, 1, 5/2, 10, 20]) #t2/t3

abc = (np.sqrt(tr)+(1/np.sqrt(tr)))*(np.sqrt(tr)+(1/np.sqrt(tr))+2) #critical ab

b = np.array([0.1, 1, 10]) #beta nondimensionalised
ab = np.array([1, 8, 9, 10, 11, 15]) #alpha nondim
N=np.array([11,12,13,14,15,16,17,18])
for Ni in N:
    for t2i in [2]:#t2:
        for t3i in [40]:#t2i/tr:
            for bi in [1.0]:#b:
                abci = (np.sqrt(t2i/t3i)+(1/np.sqrt(t2i/t3i)))*(np.sqrt(t2i/t3i)+(1/np.sqrt(t2i/t3i))+2)
                for abi in [31.4, 33.4]:#np.arange(abci-4, abci+5, 2):
                    ai = abi/bi
                    perturbation = 'steady_state'

                    if perturbation=='steady_state':
                        Dr0 = 0 #make sure there's no extension in the steady state case
                    #initial condition without the zeroth point r_0 where bead is attached
                    xinit = np.zeros(3*Ni); xinit[:Ni] += l0*np.arange(1, Ni+1, 1); 
                    xinit[Ni:-Ni] += np.ones(Ni); 
                    xinit[-Ni:] += np.random.normal(0, 0.01, Ni)

                    I = integrator(Ni, ai, bi, t2i, t3i, Dr0, lh, omega0)
                    Solution = I.integrate(t, dt, xinit, perturbation)

                    if perturbation=='osc_strain':
                        Solution = np.insert(Solution, 0, Dr0*np.sin(omega0*t), axis=1)
                    else:
                        Solution = np.insert(Solution, 0, Dr0*np.ones_like(t), axis=1)
                    print(np.sum(np.where(Solution[:,Ni-1]-xinit[Ni-1]>0,1,0)))

                    if os.path.isdir('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}'.format(perturbation, Ni ,abi, t2i, t3i)): print("dir exists")
                    else:
                        os.makedirs('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}'.format(perturbation,Ni ,abi, t2i, t3i))
                    if os.path.isfile('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.1f}.dat'.format(perturbation,Ni ,abi, t2i, t3i, bi)):
                        print("another file already exists, saving as filename_")
                        Solution.tofile('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.1f}_.dat'.format(perturbation, Ni ,abi, t2i, t3i, bi))
                    else:
                        Solution.tofile('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.1f}.dat'.format(perturbation,Ni ,abi, t2i, t3i, bi))
                    
                    parameters["beta"] = float(bi) #beta nondimensionalised
                    parameters["alpha"] = float(ai) #alpha nondim
                    parameters["t2"] = float(t2i)
                    parameters["t3"] = float(t3i) #t3 nondim
                    parameters["N"] = int(Ni)
                    
                    with open('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/parameters_{:1.1f}.json'.format(perturbation,Ni ,abi, t2i, t3i, bi), 'w') as f:
                        json.dump(parameters, f)
exit()
#below code is for running external shear on steady state
nens=50
it = [555, 384, 354, 429, 468, 299,  44,  86, 135, 447, 444, 136, 233, 511, 371, 515,  39, 400,
 429,  69, 582, 272, 583, 594, 484, 178, 442, 112, 288, 182, 345, 541, 435, 140, 269, 369,
 124, 303, 261, 246, 187, 384, 492,  16, 172, 359,  87, 170, 565, 591]
#it.tofile('lcubic_nonlinearity/{:d}/sheartimes'.format(N))

for ens in np.arange(10, nens, 1):
    xinit = np.fromfile("steady_state/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution{:d}.dat".format(N, a, t2, t3, ens//10))
    xinit = xinit.reshape(len(xinit)//(3*N +1),3*N+1)
    I = integrator()
    Solution = I.integrate(t,xinit[-it[ens], 1:])
    Solution = np.insert(Solution, 0, Dr0*np.ones(len(t)),axis=1)
    print(np.sum(np.where(Solution[:,N-1]-xinit[-it[ens], 1:][N-1]>0,1,0)))
    Solution = np.insert(Solution, 0, xinit[:-it[ens]], axis=0)
    if os.path.isdir('lcubic_nonlinearity/{:d}/{:1.1f}/{:1.1f}/{:1.1f}'.format(N ,a, t2, t3)): print("dir exists")
    else:
        os.makedirs('lcubic_nonlinearity/{:d}/{:1.1f}/{:1.1f}/{:1.1f}'.format(N ,a, t2, t3))
    if os.path.isfile('lcubic_nonlinearity/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution{:d}.dat'.format(N ,a, t2, t3, ens)):
        print("anothe file already exists, saving as filename_")
        Solution.tofile('lcubic_nonlinearity/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution{:d}_.dat'.format(N ,a, t2, t3, ens))
    else:
        Solution.tofile('lcubic_nonlinearity/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution{:d}.dat'.format(N ,a, t2, t3, ens))
with open('lcubic_nonlinearity/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/parameters.json'.format(N ,a, t2, t3), 'w') as f:
    json.dump(parameters, f)