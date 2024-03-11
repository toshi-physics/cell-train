import json, argparse, os
from turtle import width
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, TextBox
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
import matplotlib.animation as animation
from matplotlib import markers, rcParams
import numpy as np
import seaborn as sns
#plt.style.use("./tplot.mplstyle")

from scipy.optimize import curve_fit
def decsine(x,w,p,b,a):
    return a*np.exp(-b*x)*np.sin(x*w+p)

N=75
ab=10.0
t2=20.0
t3=50.0
b=1.0
a=10.0
l=1
omega0s = np.arange(0.01, 0.27, 0.01)
#omega0s = np.append(omega0s, np.array([0.015,0.025,0.035,0.045,0.046,0.047,0.048,0.049,0.050,0.051,0.052,0.053,0.054,0.055,0.065]))
#omega0s = np.append(omega0s, np.array([0.081, 0.082, 0.075]))
plensm = []
for N in [40,49,50,51,60,75,80,100]:
    print(N)
    with open("osc_strain/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/parameters_{:1.2f}.json".format(N, ab, t2, t3, omega0s[0])) as jsonFile:
        parameters = json.load(jsonFile)
        
    dt=parameters["timestep"]
    tf=parameters["tf"]
    times= np.arange(0, tf, dt)

    Sols=[]

    for omega0 in omega0s:
        Sols.append(np.fromfile("osc_strain/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.2f}.dat".format(N, ab, t2, t3, omega0)))
        Sols[-1]=Sols[-1].reshape(len(times), 3*N+1)

    #find the value of max amplitude for the last 50 timepoints
    Sols = np.asarray(Sols)
    #to find displacement, subtract each r_i from its original position
    ufields = Sols[:,5000:,:N+1]-np.arange(0,N+1,1)[np.newaxis,np.newaxis,:] #subtract each r_i from its initial position
    ufields -= np.mean(ufields, axis=1)[:,np.newaxis,:]
    fftufields = np.fft.rfft(ufields, axis=1)
    qs = np.fft.rfftfreq(len(ufields[0,:,0]), dt)
    fftmags = abs(fftufields)
    fftBufields = np.zeros_like(fftufields)
    for i in np.arange(len(omega0s)):
        fftBufields[i,np.where(qs*2*np.pi//omega0s[i]==1)[0][0],:]+=fftufields[i,np.where(qs*2*np.pi//omega0s[i]==1)[0][0],:]
        #plt.plot(qs*2*np.pi, abs(fftBufields[i,:,1]))
    #plt.show()
    #low pass filter, exclude the first mode and zero mode
    Bufield = np.fft.irfft(fftBufields, axis=1)

    plens = np.zeros(len(omega0s))
    bs = np.zeros((len(omega0s),500))
    for i in np.arange(0,len(omega0s),1):
        for t in np.arange(50):
            popt, pcov = curve_fit(decsine, np.arange(0,N+1,1), Bufield[i,-t,:], maxfev=5000)
            bs[i,t]+=popt[2]
        #print(popt)
        #plt.scatter(np.arange(0,N+1,1), Bufield[i,-1,:])
        #plt.plot(np.arange(0, N+1, 1), decsine(np.arange(0, N+1, 1),*popt),linestyle='dashed')
        #plt.show()

    plen = 20/np.mean(bs,axis=1)
    pstd = 20*np.std(bs,axis=1)/(plen*plen)
    plensm.append(plen)

plensm=np.asarray(plensm)

plt.errorbar(omega0s[8:]*12, np.mean(plensm[:,8:],axis=0), yerr=np.std(plensm[:,8:],axis=0),marker='o',linestyle='')
plt.show()
exit()
for i in np.arange(len(omega0s)):
    print(np.where(abs(Bufield[i,-1,:])/np.max(abs(Bufield[i,-1,:])) < 1/np.e))
    plt.plot(np.arange(0, N+1, 1), Bufield[i,-1])# , marker='o', color='red')
    plt.show()
plt.axvline(x=0.58, linestyle='dashed', color='black')
plt.xlabel(r'Frequency ($Hr^-1$)')
plt.ylabel('Normalised Oscillation Amplitude')
plt.show()