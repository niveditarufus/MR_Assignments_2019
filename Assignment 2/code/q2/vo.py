import numpy as np
import cv2
import os
from fun_ran import *

def Scale(f, frame_id):
      x_pre, y_pre, z_pre = f[frame_id-1][3], f[frame_id-1][7], f[frame_id-1][11]
      x, y, z = f[frame_id][3], f[frame_id][7], f[frame_id][11]
      scale = np.sqrt((x-x_pre)**2 + (y-y_pre)**2 + (z-z_pre)**2)
      return scale
      
def featureTracking(img_1, img_2, p1):
    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None)
    st = st.reshape(st.shape[0])
    p1 = p1[st==1]
    p2 = p2[st==1]
    return p1,p2

def extract_sift_keypoints(im):
    sift = cv2.xfeatures2d.SIFT_create()
    key_points = sift.detect(im)
    pts = np.array([x.pt for x in key_points],dtype=np.float32)
    return pts

def get_ground_truth():
    file = 'ground-truth.txt'
    return np.genfromtxt(file, delimiter=' ',dtype=None)

def getImages(i):
    return cv2.imread('./sample_images/'+str(i)+'.png')

def get_transformation_matrix(R,t):
    M = np.empty((4, 4))
    M[:3, :3] = R
    M[:3, 3] = t.T
    M[3, :] = [0, 0, 0, 1]
    return M

ground_truth =get_ground_truth()
img_1 = getImages(0)
img_2 = getImages(1)
p1 = extract_sift_keypoints(img_1)
p1, p2   = featureTracking(img_1, img_2, p1)

K = np.array([[7.215377000000e+02,0.000000000000e+00,6.095593000000e+02],
              [0.000000000000e+00,7.215377000000e+02,1.728540000000e+02],
              [0.000000000000e+00,0.000000000000e+00,1.000000000000e+00]])

E, mask = cv2.findEssentialMat(p2, p1, K, cv2.RANSAC,0.999,1.0);
N = len(p1)
p1new = np.ones((N,3))
p2new = np.ones((N,3))
p1new[:,:-1] = p1
p2new[:,:-1] = p2

# F = F_from_ransac(p2new.T, p1new.T, RansacModel())
# F = F[0]
# E = np.matmul(K.T,np.matmul(F,K))

x, R, t, mask = cv2.recoverPose(E, p2, p1,K);
MAX_FRAME 	  = 10 # images can be downloaded from many datasets availablr online, this sample contains only 10 images
preImage   = img_2
R_f = R
t_f = t
Tnow = get_transformation_matrix(R_f, t_f)
Tnow = Tnow[:3,:]
Tnow = Tnow.ravel()
outfile = open("output.txt","w")

l = str(' '.join(map(str, Tnow)))
outfile.write(l + os.linesep)

for numFrame in range(1, MAX_FRAME):
    print(numFrame)
    preFeature = extract_sift_keypoints(preImage)
    curImage = getImages(numFrame)
    preFeature, curFeature = featureTracking(preImage, curImage, preFeature)
    E, mask = cv2.findEssentialMat(curFeature, preFeature, K, cv2.RANSAC,0.999,1.0); 
    N = len(preFeature)
    p1new = np.ones((N,3))
    p2new = np.ones((N,3))
    p1new[:,:-1] = preFeature
    p2new[:,:-1] = curFeature
    # F = F_from_ransac(p2new.T, p1new.T, RansacModel())
    # F = F[0]
    # E = np.matmul(K.T,np.matmul(F,K))

    _, R, t, mask = cv2.recoverPose(E, curFeature, preFeature, K);
    absolute_scale = Scale(ground_truth, numFrame)

    if absolute_scale > 0.1:  
        t_f = t_f + absolute_scale*R_f.dot(t)
        R_f = R.dot(R_f)

    Tnow = get_transformation_matrix(R_f, t_f)
    Tnow = Tnow[:3,:]
    Tnow = Tnow.ravel()
    l = str(' '.join(map(str, Tnow)))
    outfile.write(l + os.linesep)

    preImage = curImage
    preFeature = curFeature
#for plotting follow instructions in the given pdf file.