import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math

def samples_sum(mu, sigma):
	samples = np.sum(np.random.uniform(-sigma,sigma,12))
	return samples + mu

def motion_model(u,x,a):
	
	u_dash1 = u[0] + samples_sum(0, (a[0]*abs(u[0])+(a[1]*u[2])))
	u_dash2 = u[1] + samples_sum(0, (a[0]*abs(u[1])+(a[1]*u[2])))
	u_dash3 = u[2] + samples_sum(0, (a[3]*( (abs(u[1]) + abs(u[0])) +(a[2]*u[2]))))

	x_dash1 = x[0] + (u_dash3*math.cos(x[2] + u_dash1))
	x_dash2 = x[1] + (u_dash3*math.sin(x[2] + u_dash1))
	x_dash3 = x[2] + u_dash1 +u_dash2

	return (np.array([x_dash1, x_dash2, x_dash3]))

x= [2.0,4.0,0]
u= [math.pi/2,0,1]
a= [.1,.1,.01,.01]

x_dash = np.zeros([5000,3])

for i in xrange(1,5000):
	x_dash[i,:] = motion_model(u,x,a)

plt.plot(x[0], x[1], "bo")
plt.plot(x_dash[:,0], x_dash[:,1], "r,")
plt.show()