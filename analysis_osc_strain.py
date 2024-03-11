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
parser.add_argument('-p','--p', help='type of perturbation: strep_strain, osc_strain, steady state')
parser.add_argument('-n','--n', help='number of cells, train length/20')
parser.add_argument('-tp','--tp', help='parimary parameter by which solutions are saved')
parser.add_argument('-ts','--ts', help='secondary parameter under primary param directory by which solutions are saved')
parser.add_argument('-tt','--tt', help='tertiary parameter under secondary param directory by which solutions are saved')
parser.add_argument('-tq','--tq',help='what number does the solution end with?, thats b')
#parser.add_argument('-ts','--timestamp', help='timestamp')
args = parser.parse_args()
#assert that data file exists
#assert os.path.isfile("lcubic_nonlinearity/{:s}/{:s}/{:s}/{:s}/solution.dat".format(args.n, args.tp, args.ts, args.tt)), 'file not found'+" solution.dat" 

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

with open("{:s}/{:s}/{:s}/{:s}/{:s}/parameters_{:s}.json".format(args.p, args.n, args.tp, args.ts, args.tt, args.tq)) as jsonFile:
	parameters = json.load(jsonFile)
l =parameters["l"]
l0=parameters["l0"]/l #eqbm rest length nondim
b = parameters["beta"] #beta nondimensionalised
a = parameters["alpha"]/l #alpha nondim
g = parameters["gamma"]
t1 = parameters["t1"]
t2 = parameters["t2"]/t1 #t2 nondim
t3 = parameters["t3"]/t1 #t3 nondim
N = int(args.n)       #number of cells (+1 is number of vertices)
Dr0 = parameters["Dr0"] #displacement of first vertex attached to the bead
tf = parameters["tf"]/t1
dt = parameters["timestep"]/t1
times= np.arange(0, 1*(tf+dt), dt)
times= np.arange(0, 1*(tf), dt)
omega0 = np.pi*2/parameters["T0"]/dt

Sol=np.fromfile("{:s}/{:s}/{:s}/{:s}/{:s}/solution_{:s}.dat".format(args.p, args.n, args.tp, args.ts, args.tt, args.tq))
Sol=Sol.reshape(len(times), 3*N+1)
maxmyosin = np.max(Sol[:,-N:])
minmyosin = np.min(Sol[:,-N:])
print(dt)
Tension0 = (Sol[:,1]-Sol[:,0]-Sol[:,N+1])
print("maxmyosin=",maxmyosin, "minmyosin=", minmyosin)

figpos, axpos = plt.subplots(1, figsize=(5.5,6))
axpos.plot(times*t1*5/60, 20*np.average(Sol[:,:N+1]-Sol[0,:N+1], axis=1))#, s=10)
axpos.plot(times*t1/12,-13.6*(1-np.exp(-times/(t2*12))), linewidth=3, color='g', linestyle='dashed')
print((20*(np.sqrt(np.heaviside(t2-t3,0) *t3+ t2*np.heaviside(t3-t2,0)))))
axpos.set_ylabel(r'u_com ($\mu m$)')
axpos.set_xlabel(r'Time (t (hrs))')
axpos.set_title(r'Mean Displacement of Center of Mass of Cell Layer')

#set figure for length graph
figl, axl = plt.subplots(1, figsize=(5.5,6))
k = np.int32((len(times)/200)); tstop = int(tf/dt)#int(592/dt) #int(tf/dt)
#this is length change
plt.contourf(20*np.arange(0*l,(N+1)*l, l), times[0:-1:k]*t1*5/60, 20*(Sol[0:-1:k,:N+1]-Sol[0,:N+1]),50)
#this is instantaeous velocity
#plt.contourf(l*np.arange(0*l,(N+1)*l, l)*20, times[1:tstop:k]*t1*5/60, (Sol[1:tstop:k,:N+1]-Sol[0:tstop:k,:N+1])*12*20/(t1*dt),50)
plt.colorbar()
axl.invert_yaxis()
#s = np.reshape(Sol[0:-1:k,1:N+1]-Sol[0:-1:k,:N], (45, N))
#axl.imshow(s, interpolation='bilinear')
axl.set_xlabel(r'Position along cell line ($\mu$m)')
#axl.set_ylabel(r'Time (t/$\tau_r$)')
axl.set_ylabel(r'Time (hr)')
axl.set_title('Kymograph of Displacement of vertices')
#axl.set_title('Velocity Kymograph')# of Displacement of vertices')


# FIND CELL NUMBER WHERE THE AMPLITUDE OF OSCILLATION IN STRAIN DROPS BELOW 10% of MAX
ampmax = np.max(Sol[:-1000, 1:N+1]-Sol[:-1000, 0:N]-1, axis=0)
print(ampmax)
print(np.where(ampmax<0.1*ampmax[0]))
figl, axl = plt.subplots(1, figsize=(5.5,6))
ablist01 = np.array([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8, 8.5])
ablist05 = np.array([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8, 8.5])
penlist01 = 20*np.array([7,7,7,7,7,8,8,8,8,9,10,10,11,12,13,19,19])
penlist05 = 20*np.array([5,2,2,2,7,8,8,8,8,9,10,10,11,12,13,19,19])

#axl.scatter(ablist0, penlist)
#axl.set_ylabel(r'Length for passive strain<10%')
#axl.set_xlabel(r'$\alpha\beta$')

#set figure for length graph
figl, axl = plt.subplots(1, figsize=(5.5,6))
k = np.int32((len(times)/200)); tstop = int(tf/dt)#int(592/dt) #int(tf/dt)
#this is passive strain
plt.contourf(l*np.arange(1*l,(N+1)*l, l)*20, times[1:tstop:k]*t1*5/60, (Sol[0:tstop:k,1:N+1]-Sol[0:tstop:k,:N]-1),50)
#this is active strain
#plt.contourf(l*np.arange(1*l,(N+1)*l, l)*20, times[1:tstop:k]*t1*5/60, (1-Sol[0:tstop:k,N+1:-N]),50)
plt.colorbar()
axl.invert_yaxis()
#s = np.reshape(Sol[0:-1:k,1:N+1]-Sol[0:-1:k,:N], (45, N))
#axl.imshow(s, interpolation='bilinear')
axl.set_xlabel(r'Position along cell line ($\mu$m)')
#axl.set_ylabel(r'Time (t/$\tau_r$)')
axl.set_ylabel(r'Time (hr)')
axl.set_title('Kymograph of Passive Strain')
#axl.set_title('Velocity Kymograph')# of Displacement of vertices')

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

#set figure for tension graph
figT, axT = plt.subplots(1, figsize=(5.5,6))
p=axT.plot(np.arange(0,(N+1), 1)*20,20*(Sol[-1,0:N+1]-Sol[0,0:N+1]))
axT.set_title('Displacement Profile')
axT.set_ylim(-50,50)

#add time axis
sax = figT.add_axes([0.1,0.94,0.85,0.02])
tbax = figT.add_axes([0.05, 0.93, 0.04, 0.04])
#tbax = fig.add_axes([0.5, 0.93, 0.04, 0.04])
tb = TextBox(tbax, 'time')
slT = Slider(sax, '', min(times), max(times), valinit=min(times))

#define animating function
def update(val):
    ti = (abs(times-val)).argmin()  #find index in times closest to the selected time on slider (val)
    p[0].set_data(np.arange(0,(N+1), 1)*20,20*(Sol[ti,0:N+1]-Sol[0,0:N+1]))
    tb.set_val(round(times[ti],5))  #show current time in textbox
    return p
    #return patches
slT.on_changed(update)

#myAnim = animation.FuncAnimation(figT, update, frames=np.arange(0, tf, 10), interval=10)
#myAnim.save("flagella_gif.gif", writer='imagemagick', fps=100)

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