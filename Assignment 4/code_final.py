import numpy as np
from numpy.linalg import inv
from math import *
import numpy.matlib
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
fontP = FontProperties()
fontP.set_size('small')
dataset = np.load('dataset.npz')

x = dataset['x_true'].ravel()
y = dataset['y_true'].ravel()
theta = dataset['th_true'].ravel()
ranges = dataset['r']
landmarks = dataset['l']
v = dataset['v'].ravel()
w = dataset['om'].ravel()
bearing = dataset['b']
v_var = dataset['v_var'][0,0]
w_var = dataset['om_var'][0,0]
b_var = dataset['b_var'][0,0]
r_var = dataset['r_var'][0,0]
d = dataset['d'][0,0]
cor = [] # corrected using EKF
act = [] # actual from ground truth
wheel_odom = [] #wheel odometry


def get_to_pi(x):
	x = x % (2 * np.pi)
	if x > np.pi:
		x = x - 2 * np.pi
	elif x < -np.pi:
		x += 2 * np.pi
	return x

def prediction(mu, sigma, v, w,v_var,w_var, Q, dt = 0.1):
	g = np.array([[cos(mu[2]),0],[sin(mu[2]),0],[0,1]])
	temp = np.array([[v+v_var],[w+w_var]])
	#print(Q.shape)
	g = dt*np.dot(g,temp)
	G = np.array([[1, 0, -v*dt*sin(mu[2])],
                  [0, 1, v*dt*cos(mu[2])],
                                  [0, 0, 1]])
	mu = mu + g
	L = dt * np.array([[cos(mu[2]),0],[sin(mu[2]),0],[0,1]])
	q = np.linalg.multi_dot([L,Q,L.T])
	sigma = np.dot(np.dot(G, sigma), G.T) + q
	mu[2] = get_to_pi(mu[2])
	wheel_odom.append(mu)

	return mu, sigma

def correction(mu, sigma, ranges, bearing, landmarks, d, r_var, b_var,R):
	idx = np.where (ranges != 0.0)
	H = np.empty([2,3])
	r = np.empty([2,1])
	z = np.empty(r.shape)
	j = 0
	for i in idx[0]:
		z[j] = ranges[i]
		bearing[i] = get_to_pi(bearing[i])
		z[j+1] = bearing[i]
        
		#range
		p = sqrt((landmarks[i,0] - mu[0] - d*cos(mu[2]))**2 + (landmarks[i,1] - mu[1] - d*sin(mu[2]))**2)
		r[j] = p
		dx = -(landmarks[i,0] - mu[0] - d*cos(mu[2]))/p
		dy = -(landmarks[i,1] - mu[1] - d*sin(mu[2]))/p
		dtheta = (-dx*d*sin(mu[2])) + (dy*d*cos(mu[2]))
		H[j] = dx,dy,dtheta
        
		#bearing
		p = atan2((landmarks[i,1] - mu[1] - d*sin(mu[2])), (landmarks[i,0] - mu[0] - d*cos(mu[2]))) - mu[2]
		r[j+1] = get_to_pi(p)
		s = ((landmarks[i,1] - mu[1] - d*sin(mu[2]))/(landmarks[i,0] - mu[0] - d*cos(mu[2])))**2
		s = 1/(1+s)
		t = 1/(landmarks[i,0] - mu[0] - d*cos(mu[2]))
		dx = s * (landmarks[i,1] - mu[1] - d*sin(mu[2])) * (t**2)
		dy = -s * t
		dtheta = s * (t**2) * ((-d*cos(mu[2])*(landmarks[i,0] - mu[0] - d*cos(mu[2]))) - (d*sin(mu[2])*(landmarks[i,1] - mu[1] - d*sin(mu[2])))) -1
		H[j+1] = dx,dy,dtheta
		L = np.eye(2)
		c = np.linalg.multi_dot([L,R,L.T])
		a = np.dot(np.dot(H,sigma),np.transpose(H))
		c = a + R
		c = np.matrix(c,dtype=float)
		inverse = inv(c)
		K = np.dot(np.dot(sigma,np.transpose(H)), inverse)
		c = np.dot(K,H)

	#update equations
		mu = mu + np.dot(K,(z.reshape(r.shape) - r))
		mu[2] = get_to_pi(mu[2])
		sigma = np.dot((np.eye(len(c)) - c),sigma)
		cor.append(np.array(mu))
	return mu, sigma

Q = np.diag([v_var, w_var])
R = np.diag([r_var, b_var])


mu = np.array([[x[0]],[y[0]],[theta[0]]])
sigma = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0.1]])
for k in range(1,len(x)):
	act.append(np.array([x[k], y[k], theta[k]]))

	mu, sigma = prediction(mu, sigma, v[k], w[k],v_var,w_var, Q)

	mu, sigma = correction(mu, sigma, ranges[k], bearing[k], landmarks, d, r_var, b_var,R)
	
cor = np.asarray(cor)
cor = np.reshape(cor,[cor.shape[0],cor.shape[1]])
act = np.asarray(act)
wheel_odom = np.asarray(wheel_odom)
wheel_odom = np.reshape(wheel_odom,[wheel_odom.shape[0], wheel_odom.shape[1]])

plt.scatter(landmarks[:,0],landmarks[:,1],c ='k',marker = '+',label='Landmarks')
plt.plot(act[:,0],act[:,1],'red', linewidth=1,markersize=1,label='Ground Truth')
plt.title("Ground truth")
plt.legend(prop=fontP,facecolor = 'gray')
plt.show()


plt.scatter(landmarks[:,0],landmarks[:,1],c ='k',marker = '+',label='Landmarks')
plt.plot(cor[:,0],cor[:,1],'blue', linewidth=1,markersize=1,label='EKF path')
plt.title("EKF path")
plt.legend(prop=fontP,facecolor = 'gray')
plt.show()