import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#######################################################
# Parameter Definition
#######################################################
source = [0,0,1] 			# Coordinate of Light Source
Lambda = 100				# Regularization Parameter
noiseSNR = 5; 				# Noise to signal ratio
radiusToImageRatio = 0.25	# Radius to Image dimensions ratio
sphereImageSize = 75   # Radius of the spehere to be rendered


#######################################################
# Detecting Boundary using Morphological Operations from OpenCV
#######################################################

rad = cv2.imread('im2.JPG')
rad = cv2.cvtColor(rad, cv2.COLOR_RGB2GRAY)
rad = cv2.resize(rad,None,fx=0.05,fy=0.05,interpolation=cv2.INTER_CUBIC)
radiance = np.asarray(rad)
plt.imshow(radiance)

regionOfInterest = radiance > 0
regionOfInterestRadiance = radiance > 0

boundaryMap              = np.zeros((sphereImageSize,sphereImageSize))
regionOfInterestRadiance = radiance > 0
kernel                   = np.ones((3,3),np.uint8)
boundaryMap              = regionOfInterestRadiance - cv2.erode(regionOfInterestRadiance.astype(np.uint8),kernel,5)
boundaryMap              = cv2.erode(cv2.dilate(boundaryMap.astype(np.uint8),kernel,3),kernel,3) 
intersectionROI          = regionOfInterest * regionOfInterestRadiance
# p,q = p * intersectionROI , q * intersectionROI
# plt.imshow(boundaryMap)
# print(sum(sum(boundaryMap)))


#######################################################
# Occluding Boundary gradients
#######################################################
# gradX, gradY       = radiance, radiance
gradX, gradY       = np.array(radiance,copy=True),np.array(radiance,copy=True)
#gradX, gradY = np.zeros(depthMap.shape),np.zeros(depthMap.shape)
gradX[:][1:-1]     = (  gradX[:][2:]   - gradX[:][:-2]) * 0.5 
gradY[1:-1][:]     = (  gradY[:-2][:]  - gradY[2:][:] ) * 0.5
# gradX = gradX * boundaryMap.astype(bool)
# gradY = gradY * boundaryMap.astype(bool)
pBoundary = np.array(gradX * boundaryMap.astype(bool),copy=True)
qBoundary = np.array(gradY * boundaryMap.astype(bool),copy=True)
pBoundary = pBoundary - pBoundary * (1 - boundaryMap)
qBoundary = qBoundary - qBoundary * (1 - boundaryMap)
# print(pBoundary)

#######################################################
# Iterative Shape from shading
#######################################################
limit = 5000
p_next,q_next = np.array(pBoundary,copy=True),np.array(qBoundary,copy=True)
p_estimated,q_estimated = np.array(pBoundary,copy=True),np.array(qBoundary,copy=True)	

for iteration in range(0,limit):
	print('Starting Iteration :', iteration+1)
	for i in range(1,pBoundary.shape[0] -1):
		for j in range(1,pBoundary.shape[1] -1):
			if regionOfInterestRadiance[i][j] == 1 :
				RadianceX,RadianceY = 0,0
				radianc = (p_estimated[i][j] * source[0] + q_estimated[i][j] * source[1] + 1) / ( ((source[0]**2+source[1]**2 + 1)**0.5) *((p_estimated[i][j]**2 + q_estimated[i][j]**2 + 1)**0.5))
				RadianceX = (p_estimated[i][j]**2 *source[0] + source[0] - q_estimated[i][j]*q_estimated[i][j]*source[1] - p_estimated[i][j])/(((source[0]**2 + source[1]**2 + 1)**0.5)*(p_estimated[i][j]**2 + (q_estimated[i][j]**2 + 1)**0.5)**3)
				RadianceY = (q_estimated[i][j]**2 *source[1] + source[1] - p_estimated[i][j]*p_estimated[i][j]*source[0] - q_estimated[i][j])/(((source[0]**2 + source[1]**2 + 1)**0.5)*(p_estimated[i][j]**2 + (q_estimated[i][j]**2 + 1)**0.5)**3)
				p_next[i][j] = 0.25*(p_estimated[i-1][j] + p_estimated[i+1][j] + p_estimated[i][j-1] + p_estimated[i][j+1]) - 1/Lambda *(radiance[i][j] - radianc)*RadianceX;
				q_next[i][j] = 0.25*(q_estimated[i-1][j] + q_estimated[i+1][j] + q_estimated[i][j-1] + q_estimated[i][j+1]) - 1/Lambda *(radiance[i][j] - radianc)*RadianceY;
	p_estimated = (p_next*regionOfInterestRadiance*(1-boundaryMap.astype(bool))) + pBoundary*boundaryMap*regionOfInterestRadiance
	q_estimated = (q_next*regionOfInterestRadiance*(1-boundaryMap.astype(bool))) + qBoundary*boundaryMap*regionOfInterestRadiance

# print(p_estimated[sphereImageSize/2])

#######################################################
# Depth Retrieval 
#######################################################
limit = 5000
Z_p   = np.zeros(p_estimated.shape)
Z     = np.zeros(p_estimated.shape)
p_x,q_y = np.array(p_estimated,copy=True),np.array(q_estimated,copy=True)
p_x[:][1:-1] = ( p_estimated[:][2:] - p_estimated[:][:-2]);
q_y[1:-1][:] = ( q_estimated[:-2][:] - q_estimated[2:][:]);
for iteration in range(0,limit):
	for i in range(1,p_estimated.shape[0]-1):
		for j in range(1,p_estimated.shape[1]-1):
			if regionOfInterestRadiance[i][j] == 1 :
				Z[i][j] = 0.25*( Z_p[i-1][j] + Z_p[i+1][j] + Z_p[i][j-1] + Z_p[i][j+1]) + abs(p_x[i][j]) + abs(q_y[i][j])
	Z_p = regionOfInterestRadiance*Z;

Z_estimated = Z * regionOfInterestRadiance
Z_estimated = 100*Z_estimated/np.amax(Z_estimated)
#Z_estimated = Z_estimated

print(np.amax(radiance))
print(np.amin(radiance))

# print(q_y[sphereImageSize/2])
# print(p_x[sphereImageSize/2])
# print(p_x[sphereImageSize/2])
#######################################################
# Visualization of the Depth
#######################################################

plt.imshow(Z_estimated)
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.set_xlim3d(0,sphereImageSize)
# ax.set_ylim3d(0,sphereImageSize)
# ax.set_zlim3d(0,sphereImageSize)
[rows,cols] = np.meshgrid(range(0,rad.shape[1]),range(0,rad.shape[0]))
surf = ax.plot_surface(rows, cols, Z_estimated, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=1, aspect=5)
plt.show()
