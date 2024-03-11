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

Ni=np.array([15,20,30,40,50,60,75,100])
modes=np.array([1,1,2,2,3,3,4,6])
xmid = Ni/(2*modes)
xmid = [7,10,7,10,8,10,9,8]
qs=modes*np.pi/Ni
natfreqs = np.array([0.777,0.638,0.777,0.638,0.725,0.638,0.669,0.770]) #from mathematica
#maxargs = [102,42,102,84,94,87,94]
maxargs = [97, 40, 97,79,90,79,83,90]
ab=10.0
t2=20.0
t3=50.0
b=1.0
a=10.0
l=1

omega0s = np.arange(0.01, 0.27, 0.01)
#omega0s = np.append(omega0s, np.arange(0.049,0.064,0.002))
#omega0s = np.sort(omega0s)
omega20s = np.append(omega0s, np.array([0.015,0.025,0.035,0.045,0.046,0.047,0.048,0.049,0.050,0.051,0.052,0.053,0.054,0.055,0.065]))

peaksN = []
maxdispN=[]
dts=[0.1,0.05,0.1,0.1,0.1,0.1,0.1,0.1]
j=0
for N in Ni:
	print(N)
	Sols=[]
	for omega0 in omega0s:
		Sols.append(np.fromfile("osc_strain/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.2f}.dat".format(N, ab, t2, t3, omega0)))
		Sols[-1]=Sols[-1].reshape(len(Sols[-1])//(3*N+1), 3*N+1)
		Sols[-1]=Sols[-1][5000:,:]
		#plt.plot(Sols[-1][-1000:,:N+1]-np.arange(0,N+1)[np.newaxis,:])
		#plt.show()
	
	#find the value of max amplitude for the last 50 timepoints
	Sols = np.asarray(Sols)
	#to find displacement, subtract each r_i from its original position
	#displacement = np.sum(Sols[:,:,:N+1]-np.arange(0,N+1,1)[np.newaxis,np.newaxis,:], axis=2) #subtract each r_i from its initial position
	displacement = Sols[:,:,:N+1]-np.arange(0,N+1,1)[np.newaxis,np.newaxis,:]
	#if (xmid[j]-int(xmid[j])) != 0: #if the number of cells is odd and there is no middle point, take avg of the two points in the middle
	#	displacement[:,:,int(xmid[j])] = (displacement[:,:,int(xmid[j])]+displacement[:,:,int(xmid[j]+1)])/2
	dispfft = np.fft.rfft(displacement[:,:,int(xmid[j])], axis=1)
	omegafft = np.fft.rfftfreq(len(displacement[0,:]), dts[j])
	fftmags = abs(dispfft); 
	Ampfft = np.copy(dispfft)
	peaks = np.zeros_like(omega0s)
	for i in np.arange(0,len(omega0s),1):
		#plt.plot(omegafft*24*np.pi,fftmags[i]/np.max(fftmags[i]))
		peaks[i] += omegafft[np.argmax(fftmags[i])]*24*np.pi
		#print(N, np.argmax(fftmags[i]), omegafft[np.argmax(fftmags[i])]*24*np.pi)
		#Ampfft[i,maxargs[j]-15:maxargs[j]+15] += dispfft[i,maxargs[j]-15:maxargs[j]+15]
		Ampfft[i,np.argmax(fftmags[i])-5:np.argmax(fftmags[i])+5] += dispfft[i,np.argmax(fftmags[i])-5:np.argmax(fftmags[i])+5]
		#plt.plot(omegafft*24*np.pi*i,abs(Ampfft[i]))
	#plt.show()
	peaksN.append(peaks)
	idisplacement = np.fft.irfft(Ampfft, axis=1, n=len(displacement[0,:]))
	#for i in np.arange(len(omega0s)):
		#plt.plot(np.arange(len(idisplacement[i])),idisplacement[i])
	#plt.show()
	#from this maximum over all cells take maximum over all times
	maxdisplacement = np.max(idisplacement,axis=1)
	maxdisplacement /= np.max(maxdisplacement)
	maxdispN.append(maxdisplacement)
	j+=1


#import theoretical approximation from dat file (imported from mathematica)
th_displacement = np.loadtxt("analytical_approximation.dat")
#th_displacement = np.loadtxt("new_a.dat")
th_displacement /= np.max(th_displacement)
th_omega0 = np.arange(0.0, 3.01, 0.01)
#th_omega0 = np.arange(0.1, 0.901, 0.01)


#plt.plot(th_omega0, th_displacement)
#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.copper(np.linspace(0,1,len(maxdispN))))
for i in np.arange(len(maxdispN)):
	plt.plot(omega0s*12/natfreqs[i], maxdispN[i],label=Ni[i])
	#plt.axvline(x=natfreqs[i], linestyle='dashed', color='black')
plt.xlabel(r'Frequency ($hr^-1$)')
plt.ylabel('Normalised Oscillation Amplitude')
plt.legend()
#plt.savefig('resonance_plot.png')
plt.show()

for i in np.arange(len(peaksN)):
	plt.plot(omega0s*12,peaksN[i])
plt.show()