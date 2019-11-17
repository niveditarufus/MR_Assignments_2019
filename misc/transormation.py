import matplotlib.pyplot as plt
import numpy as np
import csv
import math

def get_transformation(pt,theta,T):  
    c, s = np.cos(theta), np.sin(theta)
    R = (np.array(((c,-s), (s, c))))  
    pt=pt.dot(R)+T
    return pt


data = [i.strip().split() for i in open("/home/nive/space/ais.informatik.uni-freiburg.de/laserscan.dat").readlines()]
data = np.asarray(data)
data = data.ravel()
data = [float(i) for i in data]
data = np.asarray(data)

angle = np.linspace(-math.pi/2,math.pi/2,data.size)
cos_theta = np.asarray([math.cos(i) for i in angle])
cos_theta = np.multiply(data,cos_theta)
sin_theta = np.asarray([math.sin(i) for i in angle])
sin_theta = np.multiply(data,sin_theta)
laser_points = np.column_stack((cos_theta,sin_theta))

robot_centre = np.array([1,0.5])#world
robot_theta_world = math.pi/4
laser_centre = np.array([0.2, 0.0])
laser_origin =np.array([0.0,0.0])
laser_theta_robot = math.pi
laser_centre_robot = get_transformation(laser_origin, -laser_theta_robot, laser_centre)
laser_centre_world = get_transformation(laser_centre_robot, -robot_theta_world, robot_centre)
for x in xrange(0,len(laser_points)-1):
    laser_points_robot=get_transformation(laser_points[x,:], -laser_theta_robot, laser_centre)
    laser_points[x,:]=get_transformation(laser_points_robot, -robot_theta_world, robot_centre)
plt.plot(robot_centre[0],robot_centre[1],'bo')
plt.plot(laser_centre_world[0], laser_centre_world[1],'ro')
plt.plot(laser_points[:,0],laser_points[:,1],'ko')
plt.show()