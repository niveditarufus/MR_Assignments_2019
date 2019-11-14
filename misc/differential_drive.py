import numpy as np
import math
import sys

def diff_drive(x, y, theta, v_l, v_r, t, l):
	if (v_l==v_r):
		r=0
	else:
		r = l/2*(v_r + v_l)/(v_r - v_l)

	icc = np.array([x - (r * math.sin(theta)), y + (r * math.cos(theta))])
	w = (v_r - v_l)/l
	c=math.cos(w*t)
	s=math.sin(w*t)
	drive_mat = np.array([[c,-s,0],[s,c,0],[0,0,1]])

	icc_xy = np.array([[x-icc[0]],[y-icc[1]],[theta]])
	icc = np.append(icc,[w*t])
	icc= np.reshape(icc,(3,1))
	result = drive_mat.dot(icc_xy) + icc
	print(result)


x = float(sys.argv[1])
y = float(sys.argv[2])
theta = float(sys.argv[3])
v_l = float(sys.argv[4])
v_r = float(sys.argv[5])
t = float(sys.argv[6])
l = float(sys.argv[7])
diff_drive(x, y, theta, v_l, v_r, t, l)
