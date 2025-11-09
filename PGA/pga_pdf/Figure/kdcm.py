import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def get_matrix(n, top_left, top_right, bottom_left, bottom_right):
    x = np.zeros([n+1,n+1])

    for i in range(n+1):
        row_left = top_left - i*(top_left-bottom_left)/n
        row_right = top_right - i*(top_right-bottom_right)/n
        for j in range(n+1):
            x[i][j] = row_left - j*(row_left-row_right)/n
            
    return x

def get_cm(x, y, z):
    c_m = np.zeros_like(x)
    hl = np.array([1,0,0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sl = np.array([x[i][j], y[i][j], z[i][j]])
            D = hl - sl
            d = np.linalg.norm(D)
            c_m[i][j] = d
    
    return c_m

def get_kdct_cm(x, y, z):
    c_m = np.zeros_like(x)
    hl = np.array([1,0,0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sl = np.array([x[i][j], y[i][j], z[i][j]])
            # KDCT
            D = hl - sl
            d = np.linalg.norm(D)
            D = D/d
            slc = sl + 0.7071*D
            
            Dc = hl - slc
            dc = np.linalg.norm(Dc)
            c_m[i][j] = dc
    
    return c_m

def get_kdcr_cm(x, y, z):
    c_m = np.zeros_like(x)
    hl = np.array([1,0,0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            sl = np.array([x[i][j], y[i][j], z[i][j]])
            # KDCT
            idx = np.argsort(sl)[::-1]
            
            tmp = sl[idx[0]]
            k = 0
            while idx[k] != 0:
                sl[idx[k]] = sl[idx[k+1]]
                k += 1
            sl[idx[k]] = tmp
            
            D = hl - sl
            d = np.linalg.norm(D)
            c_m[i][j] = d
    
    return c_m





n = 100

fig = plt.figure(figsize=(8,4.5))
ax = mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# ax = mplot3d.Axes3D(fig)
#ax = fig.add_subplot(projection='3d')
ax.view_init(elev=25, azim=25)
ax.tick_params(labelsize=20)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.xaxis._axinfo["grid"]['linestyle'] = ":"
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis._axinfo["grid"]['linestyle'] = ":"
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis._axinfo["grid"]['linestyle'] = ":"

x_p = get_matrix(n, 1.0, 0.5, 0.5, 1/3)
y_p = get_matrix(n, 0, 0, 0.5, 1/3)
z_p = 1 - x_p - y_p

x_n1 = get_matrix(n, 0,0.5, 0,1/3)
y_n1 = get_matrix(n, 1,0.5, 0.5,1/3)
z_n1 = 1 - x_n1 - y_n1

x_n2 = get_matrix(n, 0,0, 0.5,1/3)
y_n2 = get_matrix(n, 0,0.5, 0,1/3)
z_n2 = 1 - x_n2 - y_n2

c_m = get_cm(x_p, y_p, z_p)
c_m1 = get_kdct_cm(x_n1, y_n1, z_n1)
c_m2 = get_kdct_cm(x_n2, y_n2, z_n2)

#c_max = np.max(c_m)
#c_m = c_m / c_max
#c_m1 = c_m1 / c_max
#c_m2 = c_m2 / c_max

cmap = plt.cm.get_cmap("jet")
ax.plot_surface(z_p, y_p, x_p, facecolors=cmap(c_m), cmap=cmap)
ax.plot_surface(z_n1, y_n1, x_n1, facecolors=cmap(c_m1), cmap=cmap)
ax.plot_surface(z_n2, y_n2, x_n2, facecolors=cmap(c_m2), cmap=cmap)
ax.plot_wireframe(np.array([[0,0],[0.5,1/3]]), np.array([[0,0.5],[0,1/3]]), np.array([[1,0.5],[0.5,1/3]]), color='r', linewidth=4)

mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_array(c_m)
cb = fig.colorbar(mappable, shrink=0.8, aspect=10)
cb.ax.tick_params(labelsize=20)

plt.savefig("kdctm.png")





fig = plt.figure(figsize=(8,4.5))
ax = mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
# ax = mplot3d.Axes3D(fig)
ax.view_init(elev=25, azim=25)
ax.tick_params(labelsize=20)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.xaxis._axinfo["grid"]['linestyle'] = ":"
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.yaxis._axinfo["grid"]['linestyle'] = ":"
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis._axinfo["grid"]['linestyle'] = ":"

x_p = get_matrix(n, 1.0, 0.5, 0.5, 1/3)
y_p = get_matrix(n, 0, 0, 0.5, 1/3)
z_p = 1 - x_p - y_p

x_n1 = get_matrix(n, 0,0.5, 0,1/3)
y_n1 = get_matrix(n, 1,0.5, 0.5,1/3)
z_n1 = 1 - x_n1 - y_n1

x_n2 = get_matrix(n, 0,0, 0.5,1/3)
y_n2 = get_matrix(n, 0,0.5, 0,1/3)
z_n2 = 1 - x_n2 - y_n2

c_m = get_cm(x_p, y_p, z_p)
c_m1 = get_kdcr_cm(x_n1, y_n1, z_n1)
c_m2 = get_kdcr_cm(x_n2, y_n2, z_n2)

#c_max = np.max(c_m)
#c_m = c_m / c_max
#c_m1 = c_m1 / c_max
#c_m2 = c_m2 / c_max

cmap = plt.cm.get_cmap("jet")
ax.plot_surface(z_p, y_p, x_p, facecolors=cmap(c_m), cmap=cmap)
ax.plot_surface(z_n1, y_n1, x_n1, facecolors=cmap(c_m1), cmap=cmap)
ax.plot_surface(z_n2, y_n2, x_n2, facecolors=cmap(c_m2), cmap=cmap)
ax.plot_wireframe(np.array([[0,0],[0.5,0.33]]), np.array([[0,0.5],[0,0.33]]), np.array([[1,0.5],[0.5,0.33]]), color='r', linewidth=4)

mappable = plt.cm.ScalarMappable(cmap=cmap)
mappable.set_array(c_m)
cb = fig.colorbar(mappable, shrink=0.8, aspect=10)
cb.ax.tick_params(labelsize=20)

plt.savefig("kdcrm.png")
