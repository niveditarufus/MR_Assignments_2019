import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math

def mag(V):
  return math.sqrt(sum([x*x for x in V]))

def n(V):
  v_m = mag(V)
  return [ vi/v_m for vi in V]

def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):  
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1)
    return (a,b,c)

def get_homography(uv, w):
	A = []
	for i in range(4):
		x,y = w[i][0], w[i][1]
		u,v = uv[i][0], uv[i][1]
		A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
		A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
	A = np.asarray(A)
	U, S, Vh = np.linalg.svd(A)
	L = Vh[-1,:] / Vh[-1,-1]
	H = L.reshape(3, 3)
	return H

def PoseFromHomography(H):
    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = np.cross(H1, H2)

    norm = np.linalg.norm(H1)

    T = H[:, 2] / norm
    return np.mat([H1, H2, H3, T])

def getTagPose(rt, K, lam,uv):
	r = (rt.T)[:,0:3]
	t = (rt.T)[:,3]
	temp = np.matmul(r,t)
	temp1 = lam * np.matmul(r,np.linalg.inv(K))
	temp1 = np.matmul(temp1,uv)

	return(-temp + temp1.T)

uv4 = np.array([[284.56243896, 149.2925415,1],
	[373.93179321, 128.26719666,1],
	[387.53588867, 220.2270813,1],
	[281.29962158, 241.72782898,1]])
	# [428.86453247, 114.50731659,1],
	# [453.60995483, 205.22370911,1],
	# [568.3659668, 180.55757141,1],
	# [524.76373291, 92.09218597,1]])
uv = np.array([[284.56243896, 149.2925415,1],
	[373.93179321, 128.26719666,1],
	[387.53588867, 220.2270813,1],
	[281.29962158, 241.72782898,1],
	[428.86453247, 114.50731659,1],
	[453.60995483, 205.22370911,1],
	[568.3659668, 180.55757141,1],
	[524.76373291, 92.09218597,1]])

img = cv2.imread("image.png")
tag_size = 0131.5
dist = 0079.0
K = np.array([[406.952636, 0.000000, 366.184147], [0.000000, 405.671292, 244.705127], [0.000000, 0.000000, 1.000000]])

w4 = np.array([[0,0,1],
	[tag_size,0,1],
	[tag_size,-tag_size,1],
	[0,-tag_size,1]])
	# [dist+tag_size,0,1],
	# [(2*tag_size)+dist,0,1],
	# [(2*tag_size)+dist,-tag_size,1],
	# [tag_size+dist,-tag_size,1]])
w = np.array([[0,0,1],
	[tag_size,0,1],
	[tag_size,-tag_size,1],
	[0,-tag_size,1],
	[dist+tag_size,0,1],
	[(2*tag_size)+dist,0,1],
	[(2*tag_size)+dist,-tag_size,1],
	[tag_size+dist,-tag_size,1]])

H = get_homography(uv4,w4)
projx = []
projy = []
proj = []
lam = []
for x in w:
	proj.append(np.matmul(H,x.T))
proj = np.asarray(proj)
for x in range(8):
	projx.append(proj[x][0]/proj[x][2])
	projy.append(proj[x][1]/proj[x][2])
	lam.append(proj[x][2])

projx = np.asarray(projx)
projy = np.asarray(projy)
a = np.matmul(np.linalg.inv(K),H)
print('Camera Pose:')
rt = PoseFromHomography(a)
print(rt.T)
world = []
for i in range(8):	
	world.append(getTagPose(rt, K, lam[i], uv[i]).T)
world=np.asarray(world)
world = np.reshape(world,(8,3))
print("Position:")
print(world)
coeff = equation_plane(world[0][0],world[0][1],world[0][2],world[1][0],world[1][1],world[1][2],world[2][0],world[2][1],world[2][2])
print("orientation of Tag is along the following unit vector: ")
print(n(coeff))

plt.imshow(img)
plt.plot(projx,projy,'ro', label = "points estimated from homography")	
plt.plot(uv[:,0],uv[:,1],'b+',label = "actual points")
plt.legend(loc = 'lower left')
plt.show()
