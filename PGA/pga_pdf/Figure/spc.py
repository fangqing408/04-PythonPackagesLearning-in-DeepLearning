import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

fig = plt.figure(figsize=(4.5,4.5))
plt.grid(linestyle=":")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.tick_params(labelsize=20)
plt.xlim(xmin=0)
plt.ylim(ymin=0)

plt.plot([0,0.5],[1,0.5], color='#ff585d', linewidth=3)
plt.plot([0.5,1],[0.5,0], color='#41b6e6', linewidth=3)
plt.plot(0.97,0.03, 'bs', markersize=16)
plt.plot(0.03,0.97, 'ro', markersize=16)

#plt.savefig("spc2.png")

fig = plt.figure(figsize=(4.5,4.5))
#ax = mplot3d.Axes3D(fig, auto_add_to_figure=False)
#fig.add_axes(ax)
ax = mplot3d.Axes3D(fig)
ax.view_init(elev=25, azim=25)
ax.tick_params(labelsize=20)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.xaxis._axinfo["grid"]['linestyle'] = ":"
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis._axinfo["grid"]['linestyle'] = ":"
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis._axinfo["grid"]['linestyle'] = ":"

ax.plot_surface(np.array([[0,0.5],[0,1/3]]), np.array([[1,0.5],[0.5,1/3]]), np.array([[0,0],[0.5,1/3]]), color='#ff585d')
ax.plot_surface(np.array([[0,0],[0.5,1/3]]), np.array([[0,0.5],[0,1/3]]), np.array([[1,0.5],[0.5,1/3]]), color='#41b6e6')
ax.plot_surface(np.array([[1,0.5],[0.5,1/3]]), np.array([[0,0],[0.5,1/3]]), np.array([[0,0.5],[0,1/3]]), color='#CCFF99')
ax.scatter(xs=0,ys=0,zs=1.01, color='b', s=256, marker="s")
ax.scatter(xs=0,ys=1,zs=0, color='r', s=256, marker="o")
ax.scatter(xs=1,ys=0,zs=0, color='g', s=256, marker="X")
                
#plt.savefig("spc3.png")