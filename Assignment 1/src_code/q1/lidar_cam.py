import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
import cv2
import math
import matplotlib
# np.set_printoptions(threshold=np.nan)


def load_velodyne_points(points_path):
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:,:3]                # exclude reflectance values, becomes [X Y Z]
    points = points[1::5,:]              # remove every 5th point for display speed (optional)
    points = points[(points[:,0] > 5)]   # remove all points behind image plane (approximate)
    return points

def get_points_camera_frame(lidar_points,rot):
	cam_pts = []
	for point in lidar_points:
		p = np.reshape(point,(3,1))
		p = np.dot(rot,p)
		p = p - np.array([[0.27],[0.06],[-.08]])
		p = p.ravel()
		cam_pts.append(p)
	cam_pts = np.asarray(cam_pts)
	return cam_pts
def get_image_points(points, K):
	img_pts =[]
	z=[]
	for point in points:
		p = np.reshape(point,(3,1))
		p = np.dot(K,p)
		p[0]= p[0]/p[2]
		p[1]= p[1]/p[2]
		x = p[2]
		p = p.ravel()
		if p[0]<=1240 and p[1]<=375 and p[0]>=0:
			img_pts.append(p)
			# print(x)
			z.append(x[0])
	img_pts = np.asarray(img_pts)
	return img_pts,list(z)
if __name__ == '__main__':
    points = load_velodyne_points('lidar-points.bin')
    rot = np.array([[0,-1,0], [0,0,-1], [1,0,0]])
    camera_points = get_points_camera_frame(points,rot)
    K = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02],[0.000000e+00, 7.215377e+02, 1.728540e+02], [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    image_points,z = get_image_points(camera_points,K)
    cmap = matplotlib.cm.get_cmap('viridis')
    normalize = matplotlib.colors.Normalize(vmin=min(z), vmax=max(z))
    colors = [cmap(normalize(value)) for value in z]
    image = cv2.imread('image.png')

    fig, ax = plt.subplots()	
    ax.imshow(image)
    ax.scatter(list(image_points[:,0]),list(image_points[:,1]),color = colors , marker = '.')
    cax, _ = matplotlib.colorbar.make_axes(ax)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    plt.show()

