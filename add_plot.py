import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
xr = np.linspace(-5, 5, 10)
# print(xr)
x = np.row_stack((xr, xr, xr, xr, xr, xr, xr, xr, xr, xr))
y = np.column_stack((xr, xr, xr, xr, xr, xr, xr, xr, xr, xr))
z = x * y
fig = plt.figure()
ax = Axes3D(fig)
xm, ym = np.meshgrid(x, y)
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
