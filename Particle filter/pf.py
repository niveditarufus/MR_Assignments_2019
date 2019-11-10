import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def read_lm():
	data=[]
	data = [i.strip().split() for i in open("/home/nive/space/ais.informatik.uni-freiburg.de/stub/data/world.dat").readlines()]
	data = np.asarray(data)
	data = data.astype(int)
	# print(data)
	return data

def filter(observation,particles,landmarks):
	sigma = 0.2
	n=0.0
	weights = []
	id = int(observation[0]-1)
	for particle in particles:
		expectation = math.sqrt((particle[0]-landmarks[id][1])**2 + (particle[1]-landmarks[id][2])**2)
		likelihood = scipy.stats.norm(expectation,sigma).pdf(observation[1])
		weights.append(likelihood)
	weights = np.asarray(weights)
	weights =np.reshape(weights, (np.size(particles,0),1))
	n = np.sum(weights)
	weights = weights/n
	return weights


lm = read_lm()
particles = np.array([[0, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0], [-0.2, 0.1, 0.5,]])
z = np.array([1, 2.3])
weights= filter(z, particles, lm)
print(weights)
