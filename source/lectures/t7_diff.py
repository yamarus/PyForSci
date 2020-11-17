from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams["figure.dpi"]=600
import matplotlib.pyplot as plt
import numpy as np


data=np.loadtxt('t5_diff.csv',delimiter=', ',skiprows=1)
w,v = np.linalg.eig(np.cov(data.T))

line1=np.array([[v[:,0][0],-v[:,0][0]],[v[:,0][1],-v[:,0][1]]])*w[0]**0.5*2
line2=np.array([[v[:,1][0],-v[:,1][0]],[v[:,1][1],-v[:,1][1]]])*w[1]**0.5*2

ax1=plt.subplot(222)
bins=np.linspace(-80,80,51)
plt.hist(data[:,0],bins,color='red')
plt.ylim(0,60)
ax1.set_aspect(1.4)
plt.text(50,45,r'$x^{fin}, \AA$',color='red')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.tick_params( bottom=False,)
ax1.set_yticklabels(['',20,40,60])

ax2=plt.subplot(224, sharex = ax1)
plt.hist(data[:,1],bins,color='black')
plt.text(50,45,r'$y^{fin}, \AA$')
plt.ylim(0,60)

ax2.set_aspect(1.4)
plt.xticks(np.linspace(-75,75,7))
plt.xlabel(r'$\Delta y, \AA$')

plt.subplots_adjust(hspace=-0.49)

plt.subplot(121)
plt.plot(data[:,0],data[:,1], 'g.',markersize=1)
plt.plot(line1[0],line1[1],'r-',linewidth=2, label=r'$D_1$')
plt.plot(line2[0],line2[1],'k-',linewidth=2, label=r'$D_2$')
plt.xticks(np.linspace(-75,75,7))
plt.xlim(-80,80)
plt.ylim(-80,80)
ax=plt.gca()
ax.set_aspect(1.)
plt.legend(loc='upper right')
plt.xlabel(r'$\Delta x, \AA$')
plt.ylabel(r'$\Delta y, \AA$')

plt.savefig('t7_diff.pdf')
plt.show()
