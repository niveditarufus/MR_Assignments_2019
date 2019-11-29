import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math

def samples_sum(mu, sigma):
	samples = np.empty([1,1])
	for i in range(1,1000):
		samples = np.append(samples,np.sum(np.random.uniform(-sigma,sigma,12)))
	print(samples.shape)
	return samples + mu

def rejection_sampling(mu,sigma):
	samples = np.empty([1,1])
	max_density = scipy.stats.norm(mu,sigma).pdf(mu)
	space = sigma
	for i in range(1,1000):
		x=np.random.uniform(mu-space,mu+space,1)
		y=np.random.uniform(0, max_density,1)
		if(y<=scipy.stats.norm(mu,sigma).pdf(x)):
			samples = np.append(samples,x)
	print(samples.shape)
	return samples

def box_muller(mu,sigma):
	samples = np.empty([1,1])
	for i in range(1,1000):
		u1 = np.random.uniform(0,1,1)
		u2 = np.random.uniform(0,1,1)
		x = math.cos(2*math.pi*u1)*math.sqrt(-2*math.log(u2))
		x=x*2+mu
		samples = np.append(samples,x)
	print(samples.shape)
	return samples



mu=5.0
sigma=2.0
s1 = np.random.normal(mu, sigma)
samples_s1= samples_sum(mu,sigma)
samples_s2 = rejection_sampling(mu,sigma)
samples_s3 = box_muller(mu,sigma)

plt.subplot(131)
count, bins, ignored = plt.hist(samples_s1, 50, density=True)
plt.plot(bins, scipy.stats.norm(mu,sigma).pdf(bins),linewidth=2, color='r')

plt.subplot(132)
count, bins, ignored = plt.hist(samples_s2, 50, density=True)
plt.plot(bins,scipy.stats.norm(mu,sigma).pdf(bins),linewidth=2, color='r')

plt.subplot(133)
count, bins, ignored = plt.hist(samples_s3, 50, density=True)
plt.plot(bins,scipy.stats.norm(mu,sigma).pdf(bins),linewidth=2, color='r')

plt.show()