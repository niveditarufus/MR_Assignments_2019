import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import random

def read_data():
	data = [i.strip().split() for i in open("./data/sensor_data.dat").readlines()]
	odom_data =[]
	flag = 0
	count =-1
	num = 0
	timedata=[]
	line_data=[]
	for x in data:
	    if (x[0] == 'ODOMETRY'):
	        odom_data.append(x)
	        count = count+1
	    if num != count:
	        timedata.append(line_data)
	        line_data=[]
	        num = num+1
	    if (x[0] == 'SENSOR'):
	    	line_data.append(x)
	    flag= flag+1
	    if(flag==len(data)):
	    	timedata.append(line_data)
	print odom_data
	return timedata,odom_data

def read_landmarks():
	data=[]
	data = [i.strip().split() for i in open("./data/world.dat").readlines()]
	return data

def samples_sum(mu, sigma):
	samples = np.sum(np.random.uniform(-sigma,sigma,12))
	return samples + mu

def get_particle(odom_data,x):
	new_particle=[]
	for i in range(np.size(x,0)):
		u1 = float(odom_data[0]) 
		u2 = float(odom_data[1]) 
		u3 = float(odom_data[2])
		u = np.array([u1,u2,u3])
		u_new1 = u[0] + samples_sum(0, (a[0]*abs(u[0])+(a[1]*u[2])))
		u_new2 = u[1] + samples_sum(0, (a[0]*abs(u[1])+(a[1]*u[2])))
		u_new3 = u[2] + samples_sum(0, (a[3]*( (abs(u[1]) + abs(u[0])) +(a[2]*u[2]))))
		x_new1 = x[i][0] + (u_new3*math.cos(x[i][2] + u_new1))
		x_new2 = x[i][1] + (u_new3*math.sin(x[i][2] + u_new1))
		x_new3 = x[i][2] + u_new1 +u_new2
		new_particle.append([x_new1, x_new2, x_new3])
	new_particle = np.asarray(new_particle)
	return new_particle

def filter(observation,particles,landmarks):
	sigma = 0.2
	weights = []
	id = int(observation[0]-1)
	for particle in particles:
		expectation = math.sqrt((particle[0]-landmarks[id][1])**2 + (particle[1]-landmarks[id][2])**2)
		likelihood = np.exp( - (observation[1] - expectation)**2 / (2 * sigma**2) )
		weights.append(likelihood)
	weights = np.asarray(weights)
	weights =np.reshape(weights, (np.size(particles,0),1))
	n = np.sum(weights)
	weights = np.true_divide(weights,n)
	return weights
def resample(weights,particles):
	step = 1/(len(particles))
	resampled =[]
	start = np.random.uniform(0,step)
	sum = weights[0]
	i=0
	for particle in particles:
		while(start>sum):
			i = i+1
			sum = sum + weights[i]
		resampled.append(particles[i])
		start = start + step
	resampled = np.asarray(resampled)
	return resampled

sensor_data = []
odom_data =[]
landmarks = np.empty([1,1])
sensor_data,odom_data = read_data()
landmarks = read_landmarks()
sensor_data= np.asarray(sensor_data)
odom_data = np.asarray(odom_data)
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype(float)
plt.cla()
plt.plot(landmarks[:,1],landmarks[:,2],'xb')
a= [.1,.1,.05,.05]
particles=np.random.uniform(0,1, 1500)
particles = np.reshape(particles,(500,3))
for i in range(len(odom_data)):
	print(i)
	temp_odom = np.array([odom_data[i][1],odom_data[i][2],odom_data[i][3]])
	temp_odom = temp_odom.astype(float)
	for t in (sensor_data[i]):
		new_particles = get_particle(temp_odom,particles)
		temp = np.array([t[1],t[2],t[3]])
		temp = temp.astype(float)
		weights = filter(temp, new_particles, landmarks)
		mean = np.mean(np.dot(np.transpose(weights),new_particles), axis=0)
		# print(mean.shape)
		print(mean)
		plt.plot(mean[0],mean[1],'bo')
		plt.plot(new_particles[:,0],new_particles[:,1],'r.')
		particles = resample(weights,new_particles)
plt.show()
