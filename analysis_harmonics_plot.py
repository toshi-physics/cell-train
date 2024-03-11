import numpy as np
import os
import json
from matplotlib import pyplot as plt
plt.style.use("./tplot.mplstyle")

L = np.array([140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400])
#L = np.array([140, 160, 200, 300, 500, 1000, 1500, 2000])
hs = np.array([1,   1,   1,   1,   1,    1,  1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   3,   3,   3,   3,   3,   3,   3   ,3   ,3   ,4   ,4,   4,   4,   5,   5,   5,    6 ,   6 ,   7 ,   8 ,  8  ,   9 ,   9 ,   11 , 10  , 11  ,   11,  12,  12,   12])
#hs = np.array([1, 1, 1, 2, 3, 5, 8, 11])
L_lower = 273.6
L_middle = 355.57
L_upper = 516.3
Timeperiods = np.array([3.1, 3.42, 3.8, 4.167, 4.467, 4.767, 5.167, 3.0833, 3.267, 3.45, 3.63, 3.8, 4.0, 4.16, 3.50, 4.16, 4.16, 4.0, 3.84])
ab=33.4
t2=2.0
t3=40.0
b=1.0
p='steady_state'

freql = np.array([0])
rmsdl = np.array([0])
lambdl = np.array([0])


for N in L/20:
	print(N)
	N=int(N)

	if not os.path.isfile("{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/parameters_{:1.1f}.json".format(p,int(N),ab,t2,t3,b)):
		os.rename("{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/parameters_{:1.3f}.json".format(p,int(N),ab,t2,t3,b), "{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/parameters_{:1.1f}.json".format(p,int(N),ab,t2,t3,b))
		os.rename("{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.3f}.dat".format(p,int(N),ab,t2,t3,b), "{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.1f}.dat".format(p,int(N),ab,t2,t3,b))
	
	with open("{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/parameters_{:1.1f}.json".format(p,int(N),ab,t2,t3,b)) as jsonFile:
		parameters = json.load(jsonFile)
		
	tf = parameters["tf"]
	dt = parameters["timestep"]
	times= np.arange(0, 1*(tf+dt), dt)

	Sol=np.fromfile("{:s}/{:d}/{:1.1f}/{:1.1f}/{:1.1f}/solution_{:1.1f}.dat".format(p,int(N),ab,t2,t3,b))
	Sol=Sol.reshape(len(times), 3*N+1)
    
    #take last 500 timepoints for calculating oscillation frequency through cell lengths
	freq = []
	for cell in np.linspace(1,N,N):
		lens = Sol[500:,int(cell)]-Sol[500:,int(cell)-1]
		lens -= np.mean(lens)
		ft = np.fft.rfft(lens)
		freqs = np.fft.rfftfreq(len(lens), dt)
		mags = abs(ft)
		freq.append(freqs[mags.argmax()])
	
	lens = Sol[-500:, 1:N+1] - Sol[-500:,:N]
	lens -= np.mean(lens, axis=1)[:,np.newaxis]
	if N<14:
		lens = np.append(lens, np.zeros([500, N]), axis=0)
	fts = np.fft.rfft(lens, axis=1)
	lambds = np.fft.rfftfreq(len(lens[0]), 1)
	mags = abs(fts)
	lambdl=np.append(lambdl,np.mean(lambds[mags.argmax(axis=1)]))

	freql=np.append(freql,np.mean(np.asarray(freq)))
	rmsdl=np.append(rmsdl,np.std(np.asarray(freq)))




print(lambdl)

color1 = '#ee5a31'#'#a49494'
color2 = '#5a3173'#'#397ba4'

fig, ax = plt.subplots()
tconvert = 6 #12 for tau=5 min and 6 for tau=10min
ax.errorbar(L, 1/(tconvert*freql[1:]),yerr=(1/tconvert)*(rmsdl[1:]/(freql[1:]*freql[1:])), fmt='o', c=color1)
ax1 = ax.twinx()
#ax1.scatter(L, 20/lambdl[1:], marker='o', c=color2)
ax1.scatter(L, 2*L/hs, marker='o', c=color2)
ax1.plot(L, L*L_lower/L, linestyle='dashed',c='black')
ax1.plot(L, L*L_middle/L, linestyle='dashed',c='black')
ax1.plot(L, L*L_upper/L, linestyle='dashed',c='black')
ax1.set_ylabel(r'Wavelength ($\mu m$)', c=color2)
ax.set_ylabel('Timeperiod (hr)', c=color1)
ax.set_xlabel(r'Confinement Length $L_x (\mu m)$')
#ax1.set_ylim(200, 500)
#ax.set_ylim(0, 5)
#plt.show()
plt.savefig('Figures/harmonics_plot_fft_tau_10min.svg')