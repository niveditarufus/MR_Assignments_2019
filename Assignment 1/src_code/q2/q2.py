import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import cv2
import math
from mpl_toolkits.mplot3d import axes3d, Axes3D



K = np.array([[7.2153e+02,0,6.0955e+02],[0,7.2153e+02,1.7285e+02],[0,0,1]])



img = plt.imread('image.png')
plt.imshow(img,cmap = 'gray')


# The corner of the car wheel is approximately at (800,300) ie. 
# u,v = (800,300)
# from k matrix f = 7.2153e+02
# and we know u = f*X/Z + cx
# and v = f*Y/Z +cy
# using the matrix
# cx = 6.0955e+02
# cy = 1.7285e+02
# 
# we know Y = 1.65m
# Z = f*Y/( v-cy )
# then compute
# X = (u-cx)*z/f



#from the matrix
u,v = (810,300)
f = 7.2153e+02
cx = 6.0955e+02
cy = 1.7285e+02
Y = 1.65
Z = f*Y/( v-cy )
X = (u-cx)*Z/f
print(X,Y,Z)




#generating the other points using this point
car = np.array([[0,0,0,1],
              [1.510,0,0,1],
              [1.510,-1.380,0,1],
              [0,-1.380,0,1],
              [0,0,4.1,1],
              [1.510,0,4.1,1],
              [1.510,-1.380,4.1,1],
              [0,-1.380,4.1,1]])
Rt = [[0.99,-0.087,0,X],[0.087,0.99,0,Y],[0,0,1,Z]]




result = np.zeros([8, 3])
for i in range(8):
    temp = np.matmul(K,Rt)
    result[i] = np.matmul(temp,car[i])
print(result)




for i in range(4):
    result[i] = result[i]/result[i][2]
    result[i+4] = result[i+4]/result[i+4][2]
    
print(result)




x1,y1 = [result[0][0], result[0][1]]
x2,y2 = [result[1][0], result[1][1]]
x3,y3 = [result[2][0], result[2][1]]
x4,y4 = [result[3][0], result[3][1]]
x5,y5 = [result[4][0], result[4][1]]
x6,y6 = [result[5][0], result[5][1]]
x7,y7 = [result[6][0], result[6][1]]
x8,y8 = [result[7][0], result[7][1]]




img = plt.imread('image.png')
fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(list(result[:,0]),list(result[:,1]), marker = 'o')
plt.plot([x1,x2],[y1,y2],'r-')
plt.plot([x2,x3],[y2,y3],'r-')
plt.plot([x3,x4],[y3,y4],'r-')
plt.plot([x4,x1],[y4,y1],'r-')
plt.plot([x4,x8],[y4,y8],'r-')
plt.plot([x1,x5],[y1,y5],'r-')
plt.plot([x2,x6],[y2,y6],'r-')
plt.plot([x3,x7],[y3,y7],'r-')
plt.plot([x5,x6],[y5,y6],'r-')
plt.plot([x6,x7],[y6,y7],'r-')
plt.plot([x7,x8],[y7,y8],'r-')
plt.plot([x5,x8],[y5,y8],'r-')
plt.show()






