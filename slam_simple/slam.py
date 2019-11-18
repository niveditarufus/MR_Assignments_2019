import math
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import inv
import copy

a= [.1,.1,.05,.05]

def read_data():
	sensor_readings = dict()
	data = []
	ids = []
	ranges = []
	bearings = []
	data = [i.strip().split() for i in open("./data/sensor_data.dat").readlines()]
	first = True
	timestamp = 0

	for	d in data:
		if(d[0] == "ODOMETRY"):
			sensor_readings[timestamp,'odometry'] = {'r1':float(d[1]),'t':float(d[2]),'r2':float(d[3])}
			if first: 
				first = False
			else:
				sensor_readings[timestamp,'sensor'] = {'id':ids,'range':ranges,'bearing':bearings}                
				ids=[]
				ranges = []
				bearings = []
			timestamp = timestamp +1
		if(d[0]=="SENSOR"):
			ids.append(int(d[1]))
			ranges.append(float(d[2]))
			bearings.append(float(d[3]))
		sensor_readings[timestamp-1,'sensor'] = {'id':ids, 'range':ranges, 'bearing':bearings}
	print sensor_readings
	return sensor_readings

def read_landmarks():
	landmarks = dict()
	data=[]
	data = [i.strip().split() for i in open("./data/world.dat").readlines()]
	for d in data:
		landmarks[int(d[0])]=[float(d[1]), float(d[2])]
	return landmarks

def initialise_particles(num_particles,num_landmarks):
	particles=[]
	particle = dict()
	for i in range(num_particles):
		particle['x'] = 0.0
		particle['y'] = 0.0
		particle['theta'] = 0.0
		particle['weight'] = 1/num_particles
		particle['history'] = []
		landmarks =dict()
		for i in range(num_landmarks):
			landmark = dict()
			landmark['mu'] = [0.0,0.0]
			landmark['sigma'] = np.zeros([3,3])
			landmark['observed'] =False
			landmarks[i+1] = landmark
		particle['landmarks']=landmarks
		particles.append(particle)

	return particles
def angle_diff(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

def measurement_model(particle,landmark):
	px = particle['x']
	py = particle['y']
	ptheta = particle['theta']
	lx = landmark['mu'][0]
	ly = landmark['mu'][1]
	meas_range = np.sqrt( (lx - px)**2 + (ly - py)**2 )
	meas_bearing = math.atan2(landmark['mu'][1]-particle['y'], landmark['mu'][0]-particle['x']) - particle['theta']
	h = np.array([meas_range,meas_bearing])
	H=np.zeros([2,2])
	H[0,0] = (landmark['mu'][0] - particle['x']) / h[0]
	H[0,1] = (landmark['mu'][1] - particle['y']) / h[0]
	H[1,0] = (particle['y'] - landmark['mu'][1]) / (h[0]**2)
	H[1,1] = (landmark['mu'][0] - particle['x']) / (h[0]**2)
	return h,H

def samples_sum(mu, sigma):
	samples = np.sum(np.random.uniform(-sigma,sigma,12))
	return samples + mu

def sensor_model(odometry,particles):
	u1 = float(odometry['r1']) 
	u2 = float(odometry['r2']) 
	u3 = float(odometry['t'])
	u = np.array([u1,u2,u3])
	for particle in particles:
		u_new1 = u[0] + samples_sum(0, (a[0]*abs(u[0])+(a[1]*u[2])))
		u_new2 = u[1] + samples_sum(0, (a[0]*abs(u[1])+(a[1]*u[2])))
		u_new3 = u[2] + samples_sum(0, (a[3]*( (abs(u[1]) + abs(u[0])) +(a[2]*u[2]))))
		particle['history'].append([particle['x'], particle['y']])
		particle['x'] = particle['x'] + (u_new3*math.cos(particle['theta'] + u_new1))
		particle['y'] = particle['y'] + (u_new3*math.sin(particle['theta'] + u_new1))
		particle['theta'] = particle['theta'] + u_new1 +u_new2 

def eval_sensor_model(sensor_data, particles):

	Q_t = np.array([[0.1, 0],[0, 0.1]])
	ids = sensor_data['id']
	ranges = sensor_data['range']
	count=0
	bearings = sensor_data['bearing']
	for particle in particles:
		landmarks = particle['landmarks']
		px = particle['x']
		py = particle['y']
		ptheta = particle['theta']
		for i in range(len(ids)):
			lm_id = (ids[i])
			landmark = landmarks[lm_id]
			meas_range = ranges[i]
			meas_bearing = bearings[i]
			if not landmark['observed']:
				lx = px + meas_range * np.cos(ptheta + meas_bearing)
				ly = py + meas_range * np.sin(ptheta + meas_bearing)
				landmark['mu'] = [lx, ly]
				h,H = measurement_model(particle,landmark)
				landmark['sigma']=np.dot(np.dot(H,Q_t),np.transpose(H))
				landmark['observed']= True
			else:
				h,H = measurement_model(particle,landmark)
				s=landmark['sigma']
				a=np.dot(np.dot(H,s),np.transpose(H)) + Q_t
				a= np.matrix(a,dtype=float)
				inverse=inv(a)
				K = np.dot(np.dot(s,np.transpose(H)),inverse)
				delta = [meas_range -h[0],angle_diff(meas_bearing,h[1])]
				p=landmark['mu'][0] + np.dot(K,delta)
				q=landmark['mu'][1] + np.dot(K,delta)
				landmark['mu'].append([p,q])
				landmark['sigma']=  np.dot((np.identity(2) -np.dot(K,H)),s)
				fact = 1 / np.sqrt(np.linalg.det(2* math.pi * Q_t))
				expo = -0.5 * np.dot(np.transpose(delta), np.linalg.inv(Q_t)).dot(delta)
				particle['weight'] = particle['weight'] * fact * np.exp(expo)
	normalizer=sum(particle['weight'] for particle in particles)
	for particle in particles:
		particle['weight'] = particle['weight']/normalizer

def resample(particles):
	step = 1/(len(particles))
	resampled =[]
	start = np.random.uniform(0,step)
	sum = particles[0]['weight']
	i=0
	for particle in particles:
		while(start>sum):
			i = i+1
			sum = sum + particle[i][weight]
		new = copy.deepcopy(particles[i])
		new['weight']=1/len(particles)
		resampled.append(new)
		start = start + step
	return resampled


sensor_data = read_data()
landmarks = read_landmarks()
particles = initialise_particles(100,len(landmarks))
for t in range(len(sensor_data)/2):
	print(t)
	sensor_model(sensor_data[t,'odometry'],particles)
	eval_sensor_model(sensor_data[t,'sensor'],particles)
	particles = resample(particles)
