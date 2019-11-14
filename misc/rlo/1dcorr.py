import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys
from scipy import optimize
np.set_printoptions(threshold=sys.maxsize)
r = []
rindex = []
alphavals = []
rnew = []
thetasol = []
testtheta = 30
final_rindex = []

img1 = cv2.imread('figure1.png',0)
img1 = cv2.resize(img1,(360,360))
img2 = cv2.imread('figure2.png',0)
img2 = cv2.resize(img2,(360,360))

shape = img1.shape
cx,cy = img1.shape

M = cv2.getRotationMatrix2D((cx/2,cy/2), testtheta, 1)
rotated = cv2.warpAffine(img1, M, (cx, cy))
# rotated = imutils.rotate(img1, testtheta)
# rotated = cv2.resize(rotated,(360,360))

hanw = cv2.createHanningWindow((cx,cy),cv2.CV_64F)
img1 = img1 * hanw
rotated = rotated * hanw

f1 = np.fft.fft2(img1)
fshift1 = np.fft.fftshift(f1)

f2 = np.fft.fft2(rotated)
fshift2 = np.fft.fftshift(f2)

magnitude_spectrum1 = np.log(np.abs(fshift1) +1)
polar_map1= cv2.linearPolar(magnitude_spectrum1, (cy/2,cx/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

magnitude_spectrum2 = np.log(np.abs(fshift2) +1)
polar_map2= cv2.linearPolar(magnitude_spectrum2, (cy/2,cx/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

M90 = cv2.getRotationMatrix2D((cx/2,cy/2), 90, 1)
polar_map1 = cv2.warpAffine(polar_map1, M90, (cx, cy))
polar_map2 = cv2.warpAffine(polar_map2, M90, (cx, cy))

for l1 in range(0,int(cy)):
	arr = polar_map1[l1,:]
	arrrotated = polar_map2[l1,:]

	arr = arr * np.hanning(arr.size)
	arrrotated = arrrotated * np.hanning(arrrotated.size)

	arr = np.fft.fftshift(np.fft.fft(arr))
	arrrotated = np.fft.fftshift(np.fft.fft(arrrotated))

	R = arr * np.ma.conjugate(arrrotated)
	R /= np.abs(R)

	r_l1 = np.fft.fftshift(np.fft.ifft(R).real)
	r.append(r_l1)

	result = np.argmax(r_l1)
	if (abs(((360*result)/cx) - testtheta - 180)<1):
		rindex.append(l1)
		alphavals.append(np.max(r_l1))
		# print(((360*result)/cx) - 180)

# print(rindex)
# print(alphavals)
final_rindex = [x for _,x in sorted(zip(alphavals,rindex))]
newindex = final_rindex[:180]
# print(newindex)
# plt.imshow(r, cmap = 'gray')

# img2 = cv2.warpAffine(img1, M, (cx, cy)) 


hanw = cv2.createHanningWindow((cx,cy),cv2.CV_64F)
img2 = img2 * hanw

f3 = np.fft.fft2(img2)
fshift3 = np.fft.fftshift(f3)

magnitude_spectrum3 = np.log(np.abs(fshift3) +1)
polar_map3 = cv2.linearPolar(magnitude_spectrum3, (cy/2,cx/2), cx/np.log(cx), flags=cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS)

M90 = cv2.getRotationMatrix2D((cx/2,cy/2), 90, 1)
polar_map3 = cv2.warpAffine(polar_map3, M90, (cx, cy))

H = []
for n in range(0,cx):
	if n < ((cx-1)/4):
		H.append(1)
	else:
		H.append(0)
print(H)
for l1 in newindex:
		arr = polar_map1[l1,:]
		arr = arr * np.hanning(arr.size)
		arr = np.fft.fftshift(np.fft.fft(arr))
		cmpare = polar_map3[l1,:]
		cmpare = cmpare * np.hanning(cmpare.size)
		cmpare = np.fft.fftshift(np.fft.fft(cmpare))
		R = arr * np.ma.conjugate(cmpare)
		R /= np.abs(R)
		# R = R*H
		r_l1 = np.fft.fftshift(np.fft.ifft(R).real)
		rnew.append(r_l1)
		thetasol.append(np.argmax(r_l1)-180)

rn = np.mean(rnew, axis = 0)
guessdelta = np.argmax(rn)-180
guessalpha = np.max(rn)
# print(thetasol)

n = np.arange(rn.size)
# plt.stem(n,rn)

x_data = np.asarray(n)
x_data = x_data - 180
y_data = np.asarray(rn)
# print(x_data)
# print(rn)
plt.stem(x_data,y_data)
N = cx
V = (N+1)/2
def test_func(k, alpha, delta):
    return (V/N)*(alpha * np.sinc(k + delta))/(np.sinc((k+delta)/N))
# def parabola(p, a, b):
	# return a*p**2 + b
popt, cov = optimize.curve_fit(test_func, x_data, y_data, p0=[guessalpha, -guessdelta], method = 'trf')
print(popt, cov)
plt.plot(x_data, test_func(x_data, *popt), 'r-', label='fit')
plt.legend(loc='best')
plt.show()
# plt.subplot(221),plt.imshow(img1, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])

# plt.subplot(222),plt.imshow(img2, cmap = 'gray')
# plt.title('magnitude_spectrum1'), plt.xticks([]), plt.yticks([])

# plt.subplot(223),plt.imshow(rotated, cmap = 'gray')
# plt.title('Rotated Image'), plt.xticks([]), plt.yticks([])

# plt.subplot(224),plt.imshow(polar_map3, cmap = 'gray')
# plt.title('magnitude_spectrum2'), plt.xticks([]), plt.yticks([])

plt.show()
