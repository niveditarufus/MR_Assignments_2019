import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def Cal_likelihood(m):

	x_0 = np.array([12,4])
	x_1 = np.array([5,7]) 
	d_0 = 3.9 
	d_1 = 4.5 
	var_0 = 1 
	var_1 = 1.5
	expected_d0 = math.sqrt(np.sum((m-x_0)**2))
	expected_d1 = math.sqrt(np.sum((m-x_1)**2))
	pdf0 = scipy.stats.norm.pdf(d_0,expected_d0,math.sqrt(var_0))
	pdf1 = scipy.stats.norm.pdf(d_1,expected_d1,math.sqrt(var_1))

	return pdf0,pdf1
m = np.array([6,3])
pdf0,pdf1 = Cal_likelihood(m)
x = np.arange(3, 15, 0.1)
y = np.arange(3, 15, 0.1)
xx, yy = np.meshgrid(x, y)
mv = np.column_stack((x,y))
z = np.array([Cal_likelihood([i,j]) for i,j in zip(xx.flatten(),yy.flatten())])
print(z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, z[:,0], zdir='z',  c='r' )
ax.scatter(xx, yy, z[:,1], zdir='z',  c='b' )
plt.show()

