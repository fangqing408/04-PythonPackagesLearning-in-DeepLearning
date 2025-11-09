import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = mplot3d.Axes3D(fig)
ax.view_init(elev=25, azim=25)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.xaxis._axinfo["grid"]['linestyle'] = ":"
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis._axinfo["grid"]['linestyle'] = ":"
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis._axinfo["grid"]['linestyle'] = ":"
# ax.set_zlim(0, 1)

def f(x, y):
     return np.sin(np.sqrt(x ** 2 + y ** 2))
 
x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', alpha=0.3)
cset = ax.contour(X, Y, Z, zdir='z', offset=-0.75, cmap='viridis')
ax.set_title('surface')

plt.show()





#tri = mplot3d.art3d.Poly3DCollection([[[1,0,0],[0,1,0],[0,0,1]]], facecolors='grey', cmap='viridis', alpha=0.2)
#ax.add_collection(tri)





# tri1 = np.array([[1, 0.75, 0.5, 0.5, 0.5],[0, 0.25, 0.5, 0.25, 0],[0, 0, 0, 0.25, 0.5]])

# ax.plot_trisurf(tri1[0], tri1[1], tri1[2], cmap='Reds')








# c_dict = ['r', 'g', 'b']
# x = np.random.rand(5000)
# y = np.random.rand(5000)
# z = 1 - x - y
# r = np.array([x,y,z]).T
# r[z<0] = [0,0,0]
# idx = r.argmax(1)
# c_map = np.zeros_like(idx, dtype='str')

# for i,j in enumerate(idx):
#     c_map[i] = c_dict[j]

# ax.scatter3D(x, y, z, c=c_map)






# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)


# pred_label = np.loadtxt("F:/OneDrive/Paper/TNNLS/Figure/StarLightCurves/pred_label.csv",dtype=np.float32, delimiter=",")
# true_label = np.loadtxt("F:/OneDrive/Paper/TNNLS/Figure/StarLightCurves/true_label.csv",dtype=np.float32, delimiter=",")

# x = pred_label[:,0]
# y = pred_label[:,1]
# z = pred_label[:,2]

# ax.scatter3D(x, y, z, c=z, cmap="viridis")
# ax.scatter3D(true_label[:,0], true_label[:,1], true_label[:,2], c="red")

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(pred_label[:,0], pred_label[:,1], pred_label[:,2])