
# coding: utf-8

# In[1]:


import numpy as np
from numpy.linalg import inv
from math import *
import numpy.matlib

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
print(ranges.shape)
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
	#print(temp.shape)
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
	return mu, sigma

Q = np.diag([v_var, w_var])
R = np.diag([r_var, b_var])

a = []
cor_sig =[]
cor = [] # corrected using EKS
act = [] # actual from ground truth
wee = [] # wheel odometric
mu = np.array([[x[0]],[y[0]],[theta[0]]])
sigma = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0.1]])
for k in range(1,len(x)):
	#mu, sigma = prediction(mu, sigma, v[k], w[k], Q)
	mu, sigma = prediction(mu, sigma, v[k], w[k],v_var,w_var, Q)
	predicted = mu,sigma
	pred = mu
	wee.append(pred)
	mu, sigma = correction(mu, sigma, ranges[k], bearing[k], landmarks, d, r_var, b_var,R)
	#mu, sigma = correction(mu, sigma, ranges[k], bearing[k], landmarks, d, r_var, b_var,Rdash)
	corrected = mu
	corrected_sig = sigma
	cor_sig.append(corrected_sig)
	cor.append(corrected)
	act.append(np.array([x[k], y[k], theta[k]]))
	gt = np.array([x[k], y[k], theta[k]])
	a.append(abs(corrected.ravel() - gt))
	print(abs(corrected.ravel() - gt))


# In[2]:


wee = np.asarray(wee)
cor = np.asarray(cor)
cor = cor.reshape([12608,3])
act = np.asarray(act)
print(act.shape)
wee = wee.reshape([12608,3])
print(wee.shape)
print(cor.shape)


# In[3]:


from matplotlib import pyplot as plt
plt.plot(act[:,0],act[:,1])
plt.title("Only Ground truth")


# In[4]:


plt.plot(cor[:,0],cor[:,1])
plt.title("After EKF")


# In[5]:


from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt

fontP = FontProperties()
fontP.set_size('small')


# In[6]:


# Without EKF

plt.scatter(landmarks[:,0],landmarks[:,1],c ='k',marker = '+',label='Landmarks')
plt.plot(act[:,0],act[:,1],'black', linewidth=1,markersize=1,label='Ground Truth')
plt.plot(wee[:,0],wee[:,1],'r', linewidth=1,markersize=1,label='Wheel Odometer measurements')

plt.title("Trajectory of the robot ground truth and wheel odometry")
plt.legend(prop=fontP,facecolor = 'gray')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),&nbsp; shadow=True, ncol=2)
plt.show()


# In[7]:


# With EKF
 
plt.scatter(landmarks[:,0],landmarks[:,1],c ='k',marker = '+',label='Landmarks')
plt.plot(act[:,0],act[:,1],'black', linewidth=1,markersize=2,label='Ground Truth')
plt.plot(cor[:,0],cor[:,1],'lightgreen', linewidth=1,markersize=1,label='Corrected path')

plt.title("Trajectory of the robot ground truth and corrected")
plt.legend(prop=fontP,facecolor = 'gray')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),&nbsp; shadow=True, ncol=2)
plt.show()


# In[8]:



plt.scatter(landmarks[:,0],landmarks[:,1],c ='k',marker = '+',label='Landmarks')
plt.plot(wee[:,0],wee[:,1],'r', linewidth=1,markersize=2,label='Wheel odometry')
plt.plot(cor[:,0],cor[:,1],'lightgreen', linewidth=1,markersize=1,label='Corrected path')

plt.title("Trajectory of the robot wheel odometry and corrected")
plt.legend(prop=fontP,facecolor = 'gray')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),&nbsp; shadow=True, ncol=2)
plt.show()


# In[9]:


# between EKF and Wheel odometric mesaurements
plt.scatter(landmarks[:,0],landmarks[:,1],c ='k',marker = '+',label='Landmarks')
plt.plot(act[:,0],act[:,1],'black', linewidth=1,markersize=2,label='Ground truth')
plt.plot(wee[:,0],wee[:,1],'r', linewidth=1,markersize=2,label='Wheel odometry')
plt.plot(cor[:,0],cor[:,1],'lightgreen', linewidth=1,markersize=1,label='Corrected path')

plt.title("Trajectory of the robot wheel odometry, ground truth and corrected")
plt.legend(prop=fontP,facecolor = 'gray')
#plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.00),&nbsp; shadow=True, ncol=2)
plt.show()


# In[10]:


sub = cor - wee
print(sub)


# In[11]:


gt_cor = act - cor
cor_wee = cor - wee
gt_wee = gt - wee


print('Mean square error between ground truth and corrected')
print(np.mean(gt_cor[0]**2))
print(np.mean(gt_cor[1]**2))
print(np.mean(gt_cor[2]**2))

print('Mean square error between ground truth and predicted')
print(np.mean(gt_wee[0]**2))
print(np.mean(gt_wee[1]**2))
print(np.mean(gt_wee[2]**2))


# In[12]:


x= np.linspace(0, 12608, num=12608)
plt.plot(x,abs(gt_cor[:,0]),label ='GT and Corrected')
plt.plot(x,abs(gt_wee[:,0]),label ='GT and wheel odometry')
plt.title("Absolute error between ground truth and corrected, ground truth and wheel odometry X coordinate ")
plt.legend(prop=fontP,facecolor = 'gray')
plt.show()


# In[13]:


x= np.linspace(0, 12608, num=12608)
plt.plot(x,abs(gt_cor[:,1]),label ='GT and Corrected')
plt.plot(x,abs(gt_wee[:,1]),label ='GT and wheel odometry')
plt.title("Absolute error between ground truth and corrected, ground truth and wheel odometry Y coordinate ")
plt.legend(prop=fontP,facecolor = 'gray')
plt.show()


# In[14]:


x= np.linspace(0, 12608, num=12608)
plt.plot(x,abs(gt_cor[:,2]),label ='GT and Corrected')
plt.plot(x,abs(gt_wee[:,2]),label ='GT and wheel odometry')
plt.title("Absolute error between ground truth and corrected, ground truth and wheel odometry for theta")
plt.legend(prop=fontP,facecolor = 'gray')
plt.show()

