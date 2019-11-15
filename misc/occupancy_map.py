import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def inv_sensor_model(z,x):
	if(x<z):
		return math.log((0.3)/(1-0.3))
	else:
		return math.log((0.6)/(1-0.6))

z = np.array([101, 82, 91, 112, 99, 151, 96, 85, 99, 105])
x = np.arange(0,201,10)
mapping_odds = np.zeros(len(x))
prior = math.log(0.5/(1-0.5))

for i in range(len(z)):
	for j in range(len(x)):
		if(x[j]>z[i]+20):
			continue
		mapping_odds[j] = mapping_odds[j] + inv_sensor_model(z[i],x[j]) - prior
	# print(mapping_odds[i])
mapping_odds = [1-(1/(1+math.exp(ele))) for ele in mapping_odds]
# print(mapping_odds)
plt.plot(x,mapping_odds)
plt.show()

