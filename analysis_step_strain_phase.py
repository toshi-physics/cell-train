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

#print out the current directory
#import os
#directory_path = os.getcwd()
#print("My current directory is : " + directory_path)
#folder_name = os.path.basename(directory_path)
#print("My directory name is : " + folder_name)

#parse timestamp given to plot
parser = argparse.ArgumentParser()
parser.add_argument('-n','--n', help='number of cells, train length/20')
parser.add_argument('-tp','--tp', help='parimary parameter by which solutions are saved')
parser.add_argument('-ts','--ts', help='secondary parameter under primary param directory by which solutions are saved')
parser.add_argument('-tt','--tt', help='tertiary parameter under secondary param directory by which solutions are saved')
#parser.add_argument('-ts','--timestamp', help='timestamp')
args = parser.parse_args()
it = np.loadtxt("step_strain_phase/{:s}/sheartimes.txt".format(args.n), delimiter=',')
it=np.int32(it)
#assert that data file exists
#assert os.path.isfile("step_strain_phase/{:s}/{:s}/{:s}/{:s}/solution.dat".format(args.n, args.tp, args.ts, args.tt)), 'file not found'+" solution.dat" 

def vertexpatch(ax,pos,col='k'):
    c=Circle(xy=[pos,0.5],radius=0.2,facecolor=col, edgecolor=col, zorder=2)
    ax.add_artist(c)
    return c
def cellpatch(ax,posv1,posv2,col):
    l=Line2D([posv1, posv2],[0.5,0.5],color=col, linewidth=8, zorder=1)
    ax.add_artist(l)
    return l
def draw_cells(ax,positions, myosin):
    p=np.array([])
    for i in np.arange(len(positions)-1):
        p=np.append(p, cellpatch(ax,positions[i],positions[i+1],colormap((myosin[i]-minmyosin)/(maxmyosin-minmyosin))))
    for i in range(len(positions)):
        p=np.append(p, vertexpatch(ax,positions[i]))
    return p

with open("step_strain_phase/{:s}/{:s}/{:s}/{:s}/parameters.json".format(args.n, args.tp, args.ts, args.tt)) as jsonFile:
	parameters = json.load(jsonFile)
l = parameters["l"]
l0= parameters["l0"]/l #eqbm rest length nondim
b = parameters["beta"] #beta nondimensionalised
a = parameters["alpha"]/l #alpha nondim
#g = parameters["gamma"]
t1 = parameters["t1"]
t2 = parameters["t2"]/t1 #t2 nondim
t3 = parameters["t3"]/t1 #t3nondim
N = parameters["N"]         #number of cells (+1 is number of vertices)
if N!=20:
     Dr0 = parameters["Dr0"]
else:
    Dr0 = -1.5 #displacement of first vertex attached to the bead
tf = parameters["tf"]/t1
dt = parameters["timestep"]/t1
times= np.arange(0, 1*(tf+dt), dt)

nens = 195
Sols = []
buffer = 1000
buffer2 = 14000
SSols = np.zeros((nens, int((tf+dt)//dt)+buffer, 3*N+1))
cm = sns.color_palette("Spectral", as_cmap=True)
it = int(5000/dt)+2 - it
times = np.arange(0, 2*(tf+dt), dt)
for ens in np.arange(nens):
    Sols.append(np.fromfile("step_strain_phase/{:s}/{:s}/{:s}/{:s}/solution{:d}.dat".format(args.n, args.tp, args.ts, args.tt, ens+1)))
    Sols[ens]=Sols[ens].reshape(len(Sols[ens])//(3*N +1),3*N+1)
    SSols[ens] += Sols[ens][it[ens]-buffer:,:]
    plt.plot( 20*np.average(Sols[ens][it[ens]-buffer:it[ens]+buffer2,:N+1]-Sols[ens][0,:N+1], axis=1),np.arange(-buffer, buffer2)*dt*5/60, c=cm(0.8*np.sign(np.average(Sols[ens][it[ens]-3,:N+1]-Sols[ens][it[ens]-4,:N+1]))))

plt.plot( 20*np.average(np.average(SSols[:,:buffer+buffer2,:N+1]-SSols[:,0,:N+1].reshape(nens,1,21), axis=0), axis=1),np.arange(-buffer, buffer2)*dt*5/60, c='k', lw=5)
plt.ylabel(r'Time (hr)')
plt.xlabel(r'Line Center of Mass Displacement ($\mu$m)')
plt.savefig('ens_response_flipped.svg')
plt.show()


SSols1 = SSols[:,buffer:2601,:]    #discard extra points
DiffSols = SSols1[:,40:,:] - SSols1[:,39:-1,:]
times = np.arange(0, (1200+dt), dt)
#set figure for displacement kymograph
figl, axl = plt.subplots(1, figsize=(5.5,6))
k = np.int32((len(SSols1[0,:,0])/144))
plt.contourf(20*np.arange(0*l,(N+1)*l, l), np.arange(0, len(DiffSols[0,:,1]), k)*dt*5/60, np.average(DiffSols[:,0:-1:k,:N+1], axis=0)*20/(dt*5/60), 45, cmap=cm)#/(dt*5/60),445, cmap=cm)
plt.colorbar()
axl.invert_yaxis()
axl.set_xlabel(r'Position along cell line ($\mu m$)')
axl.set_ylabel(r'Time (t/$\tau_r$)')
axl.set_ylabel(r'Time (hr)')
plt.savefig("mean_kymograph_vel_shear_20cell.svg")
plt.show()

i=17
Sol=SSols[i]
times= np.arange(0, 1*(tf+dt), dt)
maxmyosin = np.max(Sol[:,-N:])
minmyosin = np.min(Sol[:,-N:])
Tension0 = (Sol[:,1]-Sol[:,0]-Sol[:,N+1])
print("maxmyosin=",maxmyosin, "minmyosin=", minmyosin)
#set figure
fig, ax = plt.subplots(1, figsize=(5.5,6))
fig.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2)
ax.set_xlim(Sol[0,0]-3,np.max(Sol[:,N])+2)
ax.set_aspect('equal')

#colourbar
display_axes = fig.add_axes([0.05, 0.1, 0.05, 0.7])
clist = [(0, "red"), (0.25, "red"), (0.3, "orange"), (0.5, "green"), 
         (0.7, "green"), (0.75, "blue"), (1, "blue")]
colormap = plt.get_cmap('rainbow', 1024)
rvb = LinearSegmentedColormap.from_list("", clist)
cb = ColorbarBase(display_axes, cmap=colormap, orientation='vertical')
#cb.set_label('Myosin Concentration')
cb.set_ticks([0,1])
cb.set_ticklabels([np.round(minmyosin,3),np.round(maxmyosin,3)])
cb.outline.set_visible(False)
#display_axes.set_axis_off()

#draw initial condition
patches = draw_cells(ax, Sol[0,:N+1],Sol[0,-N:])

#add time axis
sax = fig.add_axes([0.1,0.94,0.85,0.02])
tbax = fig.add_axes([0.05, 0.93, 0.04, 0.04])
#tbax = fig.add_axes([0.5, 0.93, 0.04, 0.04])
tb = TextBox(tbax, 'time')
sl = Slider(sax, '', min(times), max(times), valinit=min(times))

#define animating function
def update(val):
    global patches
    ti = (abs(times-val)).argmin()  #find index in times closest to the selected time on slider (val)
    for patch in patches:
        patch.remove()
    patches = draw_cells(ax, Sol[ti,:N+1], Sol[ti,-N:])
    tb.set_val(round(times[ti],5))  #show current time in textbox
    #return patches
sl.on_changed(update)

#myAnim = animation.FuncAnimation(fig, update, frames=np.arange(0, tf, 0.1), interval=10, blit=True, repeat=True)
#myAnim.save("%s/%s_gif.gif"%(args.timestamp, args.timestamp), writer='imagemagick', fps=10)

plt.show()

exit()
###############################################################################################################


#plt.plot(np.arange(0, 900)*5/60, np.average(Sols[ens][it[ens]-100:it[ens]+800,:N+1]-Sols[ens][0,:N+1], axis=1))
#load solution data
Sol = np.fromfile("step_strain_phase/{:s}/{:s}/{:s}/{:s}/solution1.dat".format(args.n, args.tp, args.ts, args.tt))
Sol=Sol.reshape(len(times),3*N+1)
#times = np.arange(0, 2500, dt); Sol=Sol[:len(times),:]; 
maxmyosin = np.max(Sol[:,-N:])
minmyosin = np.min(Sol[:,-N:])
Tension0 = (Sol[:,1]-Sol[:,0]-Sol[:,N+1])
print("maxmyosin=",maxmyosin, "minmyosin=", minmyosin)
#Sol[0,0]=0
#set figure for tension graph
#figt, axt = plt.subplots(1, figsize=(5.5,6))
#axt.plot(times,-Tension0, 'bo')
#axt.set_title('-T_0, or force applied by the bead with time')

#set figure for tension graph but calculated from r_i's instead
#figr, axr = plt.subplots(1, figsize=(5.5,6))
#axr.plot(times[:-1],np.sum(Sol[1:,1:N+1]-Sol[:-1,1:N+1], axis=1)/N, 'bo')
#axr.set_title(r'$\sum_{i=1}^{N} \frac{r_i(t+dt)-r_i(t)}{N}$, or average displacement of "cells"')

figpos, axpos = plt.subplots(1, figsize=(5.5,6))
#for i in np.arange(1,N+1,1):
    #axpos.plot(times[:], Sol[:,i]-Sol[0,i], label="cell %f"%(i))
#axpos.loglog(times, -np.average(Sol[:,:N+1]-Sol[0,:N+1], axis=1), 'bo')
axpos.scatter(times[:]*t1, np.average(Sol[:,:N+1]-Sol[0,:N+1], axis=1), s=10)
#axpos.plot(times*t1,-1.5*(1-np.exp(-times/(20*(np.sqrt(np.heaviside(t2-t3,0) *t3+ t2*np.heaviside(t3-t2,0)))))), linewidth=3, color='r', linestyle='dashed')
#axpos.plot(times*t1,-1.44*(1-np.exp(-times/t2)), linewidth=3, color='g', linestyle='dashed')
#axpos.plot(times*t1,-1.44*(1-np.exp(-times/t3)), linewidth=3, color='b', linestyle='dashed')
axpos.plot(times*t1,0.5*Dr0*(1-np.exp(-times/(20*20*t1))), linewidth=3, color='k', linestyle='dashed')

print((20*(np.sqrt(np.heaviside(t2-t3,0) *t3+ t2*np.heaviside(t3-t2,0)))))
axpos.set_ylabel(r'u_com')
axpos.set_xlabel(r'Time (t/$\tau_r$)')
#axpos.plot(times,-1.44*(1-np.exp(-times/(499))))
#s=np.int32(tf/(3*dt))
#axpos.loglog(dt*times[s:2*s], dt*dt*times[s:2*s])
axpos.set_title(r'Mean Displacement of Center of Mass of Cell Layer')

#set figure for length graph
figl, axl = plt.subplots(1, figsize=(5.5,6))
k = np.int32((len(times)/44))
#k = np.int32((len(times)/200)); tstop = int(72/dt)
#plt.contourf(l*np.arange(0*l,(N+1)*l, l), times[0:-1:k]*t1*5/60, Sol[0:-1:k,:N+1]-Sol[0,:N+1],25)
plt.contourf(l*np.arange(0*l,(N+1)*l, l), times[0:-1:k]*t1*5/60, Sol[1:-1:k,:N+1]-Sol[0:-1:k,:N+1],25)

plt.colorbar()
axl.invert_yaxis()
#s = np.reshape(Sol[0:-1:k,1:N+1]-Sol[0:-1:k,:N], (45, N))
#axl.imshow(s, interpolation='bilinear')
axl.set_xlabel('Position along cell line')
#axl.set_ylabel(r'Time (t/$\tau_r$)')
axl.set_ylabel(r'Time (hr)')
axl.set_title('Kymograph of Displacement of vertices')

#set figure for change in rest length graph
figl, axl = plt.subplots(1, figsize=(5.5,6))
#axl.plot(times[:-1],(Sol[1:,N+1:-N]-Sol[:-1,N+1:-N])/dt)
axl.plot(times,(Sol[:,N+1:-N]-Sol[0,N+1:-N]))
#axl.set_ylim(-0.1,1)
axl.set_title('Change in rest length')

#set figure for change in myosin graph
figm, axm = plt.subplots(1, figsize=(5.5,6))
axm.plot(times, Sol[:,-N+1:])
#print((Sol[-2,-N:]-Sol[-1,-N:])/dt)
#print(Sol[-1,-N:])
#axm.set_ylim(-0.001,0.01)
axm.set_title('Myosin')

#set figure for tension graph
figT, axT = plt.subplots(1, figsize=(5.5,6))
axT.plot(times,Sol[:,1:N+1]-Sol[:,:N]-Sol[:,N+1:-N])
axT.set_title('Tensions')

#set figure
fig, ax = plt.subplots(1, figsize=(5.5,6))
fig.subplots_adjust(left=0.15, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.2)
ax.set_xlim(Sol[0,0]-3,np.max(Sol[:,N])+2)
ax.set_aspect('equal')

#colourbar
display_axes = fig.add_axes([0.05, 0.1, 0.05, 0.7])
clist = [(0, "red"), (0.25, "red"), (0.3, "orange"), (0.5, "green"), 
         (0.7, "green"), (0.75, "blue"), (1, "blue")]
colormap = plt.get_cmap('rainbow', 1024)
rvb = LinearSegmentedColormap.from_list("", clist)
cb = ColorbarBase(display_axes, cmap=colormap, orientation='vertical')
#cb.set_label('Myosin Concentration')
cb.set_ticks([0,1])
cb.set_ticklabels([np.round(minmyosin,3),np.round(maxmyosin,3)])
cb.outline.set_visible(False)
#display_axes.set_axis_off()

#draw initial condition
patches = draw_cells(ax, Sol[0,:N+1],Sol[0,-N:])

#add time axis
sax = fig.add_axes([0.1,0.94,0.85,0.02])
tbax = fig.add_axes([0.05, 0.93, 0.04, 0.04])
#tbax = fig.add_axes([0.5, 0.93, 0.04, 0.04])
tb = TextBox(tbax, 'time')
sl = Slider(sax, '', min(times), max(times), valinit=min(times))

#define animating function
def update(val):
    global patches
    ti = (abs(times-val)).argmin()  #find index in times closest to the selected time on slider (val)
    for patch in patches:
        patch.remove()
    patches = draw_cells(ax, Sol[ti,:N+1], Sol[ti,-N:])
    tb.set_val(round(times[ti],5))  #show current time in textbox
    #return patches
sl.on_changed(update)

#myAnim = animation.FuncAnimation(fig, update, frames=np.arange(0, tf, 0.1), interval=10, blit=True, repeat=True)
#myAnim.save("%s/%s_gif.gif"%(args.timestamp, args.timestamp), writer='imagemagick', fps=10)

plt.show()