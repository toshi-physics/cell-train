from cProfile import label
import os
import json, argparse
import numpy as np
from scipy.sparse import diags
from numba import jit

initParser = argparse.ArgumentParser(description='Run 1D Vertex Model simulation')
initParser.add_argument('-j','--jsonfile', help='parameter file of type json', default='parameters_Model_B_dump.json')
initParser.add_argument('-p','--perturbation', help='which kind of perturbation? osc_strain, step_strain, steady_state?')
initargs = initParser.parse_args()
if os.path.isfile("%s" %(initargs.jsonfile)):
	with open(initargs.jsonfile) as jsonFile:
		parameters = json.load(jsonFile)

l = parameters["l"]
l0= parameters["l0"]/l #eqbm rest length nondim
b = parameters["beta"] #beta nondimensionalised
a = parameters["alpha"]/l #alpha nondim
g = parameters["gamma"]
t1 = parameters["t1"]
t2 = parameters["t2"]/t1 #t2 nondim
t3 = parameters["t3"]/t1 #t3 nondim
N = parameters["N"]         #number of cells (+1 is number of vertices)
Dr0 = parameters["Dr0"] #displacement of first vertex attached to the bead
omega0 = t1*2*np.pi/parameters["T0"] #oscillation frequency of the external strain
tf = parameters["tf"]/t1
dt = parameters["timestep"]/t1
dt_dump= parameters["dump_timestep"]
t = np.arange(0, tf+dt, dt)
t_dump = np.arange(0, tf+dt_dump, dt_dump)
# lh = l-(Dr0)/N # this makes lbar = L(t)/N
lh = l0 # this makes lbar L(0)/N

class integrator:
    def __init__(self, N, a, b, t2, t3, Dr0, lh, omega0):

        self.N = N
        self.a = a
        self.b = b
        self.t2 = t2
        self.t3 = t3
        self.Dr0= Dr0
        self.lh = lh
        self.omega0 = omega0

        diag0 = -1*np.ones(3*self.N); diag0[:self.N-1]*=2
        diag0[self.N:2*self.N] *= 1/self.t2; diag0[2*self.N:] *= 1/self.t3

        diag1 = np.zeros(3*self.N-1); diag1[:self.N-1] += 1
        diagm1 = np.zeros(3*self.N-1); diagm1[:self.N-1] += 1
        diagN = np.zeros(2*self.N); diagN[:self.N] += 1
        diagNp1 = np.zeros(2*self.N-1); diagNp1[:self.N-1] -= 1

        #diagN[self.N:2*self.N] -= self.a/self.t2 

        diagm2N = np.zeros(self.N); #diagm2N[:self.N] += self.b/(self.lh*self.t3)
        diagm2Nm1 = np.zeros(self.N-1); #diagm2Nm1[:self.N-1] -= self.b/self.t3

        self.A = diags([diag0, diag1, diagm1, diagN, diagNp1, diagm2N, diagm2Nm1], [0,1,-1,self.N,self.N+1,-2*self.N,-2*self.N-1], format='csr')
        self.C = np.zeros(3*self.N); self.C[0] += self.Dr0
        self.C[self.N:2*self.N] += 1/self.t2 #const part of the rest length decay term, -1/self.t2 *(l0-l*)/l*
        #self.C[self.N-1] += (1-l0)  #lamelipodia tension is equal to a constant stall tension

    def ode(self, t,x):
        n = self.A*x + self.C
        n[self.N-1]*=0
        #nn = np.zeros_like(n); nn[2*self.N:] -= g*np.power(x[2*self.N:],3)/self.t3
        nn = np.zeros_like(n); nn[2*self.N+1:] += self.b*np.tanh((x[1:self.N]-x[:self.N-1]-self.lh)/self.lh)/self.t3
        nn[2*self.N] += self.b*np.tanh((x[0]-self.Dr0-self.lh)/self.lh)/self.t3
        nn[N:2*self.N] -= self.a*np.tanh(x[2*self.N:])/self.t2
        #n[self.N-1] *= np.heaviside(self.N*l - x[self.N-1] - n[self.N-1]*dt, 1)#position of last cell cannot be beyond a certain length, self.N*l
        return n + nn
    
    def oscode(self, t,x):
        self.C[0]=self.Dr0*np.sin(self.omega0*t); #for oscillatory, fix time dependent displacement of first end
        n = self.A*x + self.C
        n[self.N-1]*=0
        nn = np.zeros_like(n); nn[2*self.N+1:] += self.b*np.tanh((x[1:self.N]-x[0:self.N-1]-self.lh)/self.lh)/self.t3
        nn[2*self.N] += self.b*np.tanh((x[0]-self.Dr0*np.sin(self.omega0*t)-self.lh)/self.lh)/self.t3
        nn[N:2*self.N] -= self.a*np.tanh(x[2*self.N:])/self.t2
        #n[self.N-1] *= np.heaviside(self.N*l - x[self.N-1] - n[self.N-1]*dt, 1)#position of last cell cannot be beyond a certain length, self.N*l
        return n + nn
    
    def rk4step(self, f, t, x, dt):
        k1 = f( t, x)
        k2 = f( t+ dt/2,  x+ (k1*dt/2))
        k3 = f( t+ dt/2,  x+ (k2* dt/2))
        k4 = f( t+ dt,  x+ (k3* dt))
        return  x+(dt*(k1+2*k2+2*k3+k4)/6)

    def integrate(self, t, dt, dt_dump, x_init, perturbation):
        f = self.ode
        if  perturbation=='osc_strain':
            f = self.oscode
        ndump = np.round(dt_dump/dt)
        dumpSol = np.array([x_init])
        prevSol = x_init
        for i in np.arange(0,t.size-1,1):
            Sol = self.rk4step(f,t[i],prevSol,dt)
            if i%ndump == 0:
                dumpSol=np.append(dumpSol, np.array([Sol]), axis=0)
            prevSol=Sol
        return dumpSol

perturbation = initargs.perturbation

if perturbation=='steady_state':
    Dr0 = 0 #make sure there's no extension in the steady state case

#initial condition without the zeroth point r_0 where bead is attached
xinit = np.zeros(3*N); xinit[:N] += l0*np.arange(1, N+1, 1); xinit[N:-N] += np.ones(N); xinit[-N:] += np.random.normal(0, 0.01, N)

abcrit = (np.sqrt(t2/t3)+(1/np.sqrt(t2/t3)))*(np.sqrt(t2/t3)+(1/np.sqrt(t2/t3))+2)

for N in [ 300, 400 ]:
    for omega0 in np.arange(0.3,1,0.1):
        print(N, omega0, a*b)
        #print(N, a*b)
        ai=a
        abi = ai*b
        bi = b
        quart=bi
        
        if perturbation=='step_strain_extend':
            xinitall = np.fromfile("steady_state/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.1f}.dat".format(N, abi, t2, t3, bi))
            xinitall = xinitall.reshape(len(xinitall)//(3*N +1),3*N+1)
            xinit = xinitall[-1,1:]
            quart = -Dr0
        elif perturbation=='osc_strain':
            xinitall = np.fromfile("steady_state/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.1f}.dat".format(N, abi, t2, t3, bi))
            xinitall = xinitall.reshape(int(len(xinitall)/(3*N +1)),3*N+1)
            startpos = -int(2000*np.random.rand(1))
            print(startpos)
            xinit = xinitall[startpos,1:]
        else:
            xinit = np.zeros(3*N); xinit[:N] += l0*np.arange(1, N+1, 1); 
            xinit[N:-N] += np.ones(N); 
            xinit[-N:] += np.random.normal(0, 0.01, N)

        I = integrator(N, ai, bi, t2, t3, Dr0, lh, omega0)
        Solution = I.integrate(t, dt, dt_dump, xinit, perturbation)
        if perturbation=='osc_strain':
            Solution = np.insert(Solution, 0, Dr0*np.sin(omega0*t_dump), axis=1)
            quart = omega0
            parameters["tf"] = float(len(Solution[:,0])*dt_dump)
            parameters["starttime"] = float(startpos*dt)
        else:
            Solution = np.insert(Solution, 0, Dr0*np.ones_like(t_dump), axis=1)

        if perturbation=='step_strain_extend':
            Solution = np.insert(Solution, 0, xinitall[-500:], axis=0)
            parameters["tf"] = float(len(Solution[:,0])*dt)
        
        #print(np.sum(np.where(Solution[:,N-1]-xinit[N-1]>0,1,0)))

        if os.path.isdir('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}'.format(perturbation, N ,abi, t2, t3)): print("dir exists")
        else:
            os.makedirs('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}'.format(perturbation,N ,abi, t2, t3))
        if os.path.isfile('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.3f}.dat'.format(perturbation,N ,abi, t2, t3, quart)):
            print("another file already exists, saving as filename_")
            Solution.tofile('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.3f}_.dat'.format(perturbation,N ,abi, t2, t3, quart))
        else:
            Solution.tofile('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.3f}.dat'.format(perturbation,N ,abi, t2, t3, quart))

        parameters["beta"] = float(bi) #beta nondimensionalised
        parameters["alpha"] = float(ai) #alpha nondim
        parameters["T0"] = float(2*np.pi/omega0)

        with open('{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/parameters_{:1.3f}.json'.format(perturbation,N ,abi, t2, t3, quart), 'w') as f:
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