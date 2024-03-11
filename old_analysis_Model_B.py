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

alpha = [6.0, 8.0, 8.2, 9.0, 10.8, 15.0]
paramvals = [[8.0,20.0,70.0],[8.0,70.0,20.0],[10.8,20.0,70.0],[10.8,70.0,20.0],[15.0,20.0,70.0],[15.0,70.0,20.0],[6.0,20.0,20.0],[8.2,20.0,20.0],[15.0,20.0,20.0],[6.0,70.0,70.0],[8.2,70.0,70.0],[15.0,70.0,70.0], [9.0,60.0,30.0],[9.0,30.0,60.0],[9.0,15.0,30.0], [15.0,60.0,30.0],[15.0,30.0,60.0],[15.0,15.0,30.0],[6.0,30.0,60.0],[6.0,15.0,30.0]]

N=10
dt=5e-2
alpha = 15.0
tau2list=np.array([15.0,20.0,30.0,45.0,60.0,70.0])
tau3list=np.array([70.0,45.0,20.0])
meanf = []
rmsdf = []

for tau3 in tau3list:
	freql =[]
	rmsdl =[]
	for tau2 in tau2list:
		with open("cubic_nonlinearity/20/%s/%s/%s/parameters.json" % (alpha, tau2, tau3)) as jsonFile:
			parameters = json.load(jsonFile)
		tf = parameters["tf"]
		times= np.arange(0, 1*(tf+dt), dt)
		Sol = np.fromfile("cubic_nonlinearity/20/%s/%s/%s/solution.dat" % (alpha, tau2, tau3))
		Sol=Sol.reshape(len(times),3*N+1)
		freq = []
		for cell in np.linspace(1,20,20):
			lens = Sol[:,np.int(cell)]-Sol[:,np.int(cell)-1]
			lens -= np.mean(lens)
			ft = np.fft.rfft(lens)
			freqs = np.fft.rfftfreq(len(lens), dt)
			mags = abs(ft)
			freq.append(freqs[mags.argmax()])
		freql.append(np.mean(np.asarray(freq)))
		rmsdl.append(np.std(np.asarray(freq)))
	meanf.append(np.asarray(freql))
	rmsdf.append(np.asarray(rmsdl))

figpos, axpos = plt.subplots(1, figsize=(5.5,6))
import seaborn as sns
palette = sns.color_palette(None, len(tau3list))
i=0
omegafrommathematica = np.array([[0.0678,0.06127,0.05186,0.04314,0.0376,0.0348],[0.0897,0.0794,0.06596,0.05415,0.04677,0.04314],[0.14031,0.12184,0.09895,0.0794,0.0673,0.0613]])
while i<len(tau3list):
	axpos.errorbar(tau2list, meanf[i], yerr=rmsdf[i],marker='o', label=r'$\tau_3$= %1.1f' % (tau3list[i]), color=palette[i], linestyle="None", capsize=3)
	#axpos.plot(tau2list,omegafrommathematica[i]/(2*np.pi), markersize=5, color=palette[i], marker="D", linestyle="None")
	axpos.scatter(tau2list,omegafrommathematica[i]/(2*np.pi), color=palette[i], marker="D", edgecolor='k')
	i+=1
axpos.legend()
axpos.set_xlabel(r'$\tau_2$')
axpos.set_ylabel(r'$\f_{osc}$')
axpos.xaxis.label.set_fontsize(20)
axpos.yaxis.label.set_fontsize(20)
i=0
figlog, axlog = plt.subplots(1, figsize=(5.5,6))
while i<len(tau3list):
	axlog.errorbar(np.log(tau2list), np.log(meanf[i]), yerr=rmsdf[i]/meanf[i],marker='o', label=r'$\tau_3$= %1.1f' % (tau3list[i]), color=palette[i], linestyle="None", capsize=3)
	#axpos.plot(tau2list,omegafrommathematica[i]/(2*np.pi), markersize=5, color=palette[i], marker="D", linestyle="None")
	axlog.scatter(np.log(tau2list),np.log(omegafrommathematica[i]/(2*np.pi)), color=palette[i], marker="D", edgecolor='k')
	i+=1
#axlog.plot(np.log(tau2list[1:-1]), np.log(np.power(tau2list[1:-1],-0.5))-3.5, color='k')
axlog.plot(np.log(tau2list[1:-1]), np.log(np.power(tau2list[1:-1],-1.5)), color='k')
axlog.legend()
axlog.set_xlabel(r'$log(\tau_2)$')
axlog.set_ylabel(r'$log(\omega_{osc})$')
axlog.xaxis.label.set_fontsize(20)
axlog.yaxis.label.set_fontsize(20)
axlog.text(3.3,-5.4, 'Slope=-1/2')
plt.rc('legend', fontsize=15)
plt.show()