import numpy as np
from scipy.sparse import diags

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

        diagN[self.N:2*self.N] -= self.a/self.t2

        diagm2N = np.zeros(self.N); #diagm2N[:self.N] += self.b/(self.lh*self.t3)
        diagm2Nm1 = np.zeros(self.N-1); #diagm2Nm1[:self.N-1] -= self.b/self.t3

        self.A = diags([diag0, diag1, diagm1, diagN, diagNp1, diagm2N, diagm2Nm1], [0,1,-1,self.N,self.N+1,-2*self.N,-2*self.N-1], format='csr')
        self.C = np.zeros(3*self.N); self.C[0]+=self.Dr0
        self.C[self.N:2*self.N]+=1/self.t2 #const part of the rest length decay term, -1/self.t2 *(l0-l*)/l*
        #self.C[self.N-1]+=(1-l0)  #lamelipodia tension is equal to a constant stall tension

    def ode(self, t,x):
        n = self.A*x + self.C
        n[self.N-1]*=0
        #nn = np.zeros_like(n); nn[2*self.N:] -= g*np.power(x[2*self.N:],3)/self.t3
        nn = np.zeros_like(n); nn[2*self.N+1:] += self.b*np.tanh((x[1:self.N]-x[:self.N-1]-self.lh)/self.lh)/self.t3
        nn[2*self.N] += self.b*np.tanh((x[0]-self.Dr0-self.lh)/self.lh)/self.t3
        #n[self.N-1] *= np.heaviside(self.N*l - x[self.N-1] - n[self.N-1]*dt, 1)#position of last cell cannot be beyond a certain length, self.N*l
        return n + nn
    
    def oscode(self, t,x):
        self.C[0]=self.Dr0*np.sin(self.omega0*t); #for oscillatory, fix time dependent displacement of first end
        n = self.A*x + self.C
        n[self.N-1]*=0
        nn = np.zeros_like(n); nn[2*self.N+1:] += self.b*np.tanh((x[1:self.N]-x[0:self.N-1]-self.lh)/self.lh)/self.t3
        nn[2*self.N] += self.b*np.tanh((x[0]-self.Dr0*np.sin(self.omega0*t)-self.lh)/self.lh)/self.t3
        #n[self.N-1] *= np.heaviside(self.N*l - x[self.N-1] - n[self.N-1]*dt, 1)#position of last cell cannot be beyond a certain length, self.N*l
        return n + nn

    def rk4step(self, f, t, x, dt):
        k1 = f( t, x)
        k2 = f( t+ dt/2,  x+ (k1*dt/2))
        k3 = f( t+ dt/2,  x+ (k2* dt/2))
        k4 = f( t+ dt,  x+ (k3* dt))
        return  x+(dt*(k1+2*k2+2*k3+k4)/6)

    def integrate(self, t, dt, x_init, perturbation):
        f = self.ode
        if  perturbation=='osc_strain':
            f = self.oscode
        Sol = np.array([x_init])
        for i in np.arange(0,t.size-1,1):
            Sol = np.append(Sol, np.array([self.rk4step(f,t[i],Sol[i],dt)]), axis=0)
        return Sol