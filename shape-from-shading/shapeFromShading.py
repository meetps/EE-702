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
Lambda = 0.001				# Regularization Parameter
noiseSNR = 5; 				# Noise to signal ratio
radiusToImageRatio = 0.5	# Radius to Image dimensions ratio
sphereImageSize = 50     # Radius of the spehere to be rendered

#######################################################
# Rendering the 3D Surface
#######################################################
depthMap         = np.zeros((sphereImageSize,sphereImageSize))                     # Array to store depth (z) values
regionOfInterest = np.zeros((sphereImageSize,sphereImageSize))                     # Boolean flag to mark the sphere ROI
radius           = radiusToImageRatio * sphereImageSize                            # Radius of the sphere
[cols,rows] 	 = np.meshgrid(range(0,sphereImageSize),range(0,sphereImageSize))  # Meshgrid for base of computation

#Calculating the depth using z^2 = r^2 - x^2 - y^2  for each point in the depth map
for i in range(0,sphereImageSize):
	for j in range(0,sphereImageSize):
		depthMap[i][j] = radius**2 - math.pow(cols[i][j] - sphereImageSize/2 , 2) - math.pow(rows[i][j] - sphereImageSize/2 , 2);
		if(depthMap[i][j] > 0):
			regionOfInterest[i][j] = 1;

depthMap = np.sqrt(depthMap * regionOfInterest)
# print(depthMap)
# print(regionOfInterest)

#######################################################
# Calculating the x, y Gradient Fields p and q
#######################################################
p,q = np.zeros((sphereImageSize,sphereImageSize)),np.zeros((sphereImageSize,sphereImageSize))
for i in range(1,sphereImageSize-1):
	for j in range(1,sphereImageSize-1):
		p[i][j] = ( depthMap[i][j+1] - depthMap[i][j-1] )/ 2 * regionOfInterest[i][j]
		q[i][j] = ( depthMap[i-1][j] - depthMap[i+1][j] )/ 2 * regionOfInterest[i][j]

p,q = p * regionOfInterest, q * regionOfInterest
# print(p)
#######################################################
# Calculating the image radiance from gradient fields
#######################################################
radiance = np.zeros((sphereImageSize,sphereImageSize))
for i in range(0,sphereImageSize):
	for j in range(0,sphereImageSize):
		if(regionOfInterest[i][j]):
			radiance[i][j] = (p[i,j]*source[0] + q[i,j]*source[1] + 1)/(((source[0]**2+source[1]**2 + 1)**0.5)*((p[i,j]**2 + q[i,j]**2 + 1)**0.5))
			if (radiance[i][j] < 0 ):
				radiance[i][j] = 0
# print(radiance)

#######################################################
# Detecting Boundary using Morphological Operations from OpenCV
#######################################################
boundaryMap              = np.zeros((sphereImageSize,sphereImageSize))
regionOfInterestRadiance = radiance > 0
kernel                   = np.ones((3,3),np.uint8)
boundaryMap              = regionOfInterestRadiance - cv2.erode(regionOfInterestRadiance.astype(np.uint8),kernel,2)
boundaryMap              = cv2.erode(cv2.dilate(boundaryMap.astype(np.uint8),kernel,2),kernel,2) 
# print(sum(sum(boundaryMap)))

#######################################################
# Occluding Boundary gradients
#######################################################
gradX, gradY       = radiance, radiance
gradX[:][1:-1]     = (  gradX[:][2:]   - gradX[:][:-2]) * 0.5 
gradY[1:-1][:]     = (  gradY[:-2][:]  - gradY[2:][:] ) * 0.5
gradX = gradX * boundaryMap.astype(bool)
gradY = gradY * boundaryMap.astype(bool)
pBoundary = gradX * boundaryMap.astype(bool)
qBoundary = gradY * boundaryMap.astype(bool)

# print(pBoundary)

#######################################################
# Iterative Shape from shading
#######################################################
limit = 10
p_next,q_next = pBoundary,qBoundary
p_estimated,q_estimated = p_next,q_next	

for iteration in range(0,limit):
	print('Starting Iteration :', iteration+1)
	for i in range(1,pBoundary.shape[0] -1):
		for j in range(1,pBoundary.shape[1] -1):
			if regionOfInterestRadiance[i][j] == 1 :
				RadianceX,RadianceY = 0,0
				radianc = (p_estimated[i][j] * source[0] + q_estimated[i][j] * source[1] + 1) / ( ((source[0]**2+source[1]**2 + 1)**0.5) *((p_estimated[i][j]**2 + q_estimated[i][j]**2 + 1)**0.5))
				RadianceX = (p_estimated[i][j]**2 *source[0] + source[0] - q_estimated[i][j]*q_estimated[i][j]*source[1] - p_estimated[i][j])/(((source[0]**2 + source[1]**2 + 1)**0.5)*(p_estimated[i][j]**2 + (q_estimated[i][j]**2 + 1)**0.5)**3)
				RadianceY = (q_estimated[i][j]**2 *source[1] + source[1] - p_estimated[i][j]*p_estimated[i][j]*source[0] - q_estimated[i][j])/(((source[0]**2 + source[1]**2 + 1)**0.5)*(p_estimated[i][j]**2 + (q_estimated[i][j]**2 + 1)**0.5)**3)
				p_next[i][j] = 0.25*(p_estimated[i-1][j] + p_estimated[i+1][j] + p_estimated[i][j-1] + p_estimated[i][j+1]) + 1/Lambda *(radiance[i][j] - radianc)*RadianceX;
				q_next[i][j] = 0.25*(q_estimated[i-1][j] + q_estimated[i+1][j] + q_estimated[i][j-1] + q_estimated[i][j+1]) + 1/Lambda *(radiance[i][j] - radianc)*RadianceY;
	p_estimated = (p_next*regionOfInterestRadiance*(1-boundaryMap.astype(bool))) + pBoundary*boundaryMap*regionOfInterestRadiance
	q_estimated = (q_next*regionOfInterestRadiance*(1-boundaryMap.astype(bool))) + qBoundary*boundaryMap*regionOfInterestRadiance

# print(p_estimated[sphereImageSize/2])

#######################################################
# Depth Retrieval 
#######################################################
limit = 100
Z_p   = np.zeros(p_estimated.shape)
Z     = np.zeros(p_estimated.shape)
p_x,q_y = p_estimated,q_estimated
p_x[:][1:-1] = ( p_estimated[:][2:] - p_estimated[:][:-2])*0.1;
q_y[1:-1][:] = ( q_estimated[:-2][:] - q_estimated[2:][:])*0.1;
for iteration in range(0,limit):
	for i in range(1,p_estimated.shape[0]-1):
		for j in range(1,p_estimated.shape[1]-1):
			if regionOfInterestRadiance[i][j] == 1 :
				Z[i][j] = 0.5*( Z_p[i-1][j] + Z_p[i+1][j] + Z_p[i][j-1] + Z_p[i][j+1]) - abs(p_x[i][j]) - abs(q_y[i][j])
	Z_p = regionOfInterestRadiance*Z;

Z_estimated = Z * regionOfInterestRadiance
Z_estimated = -Z_estimated

# print(q_y[sphereImageSize/2])
print(Z_p[sphereImageSize/2])
# print(p_x[sphereImageSize/2])
#######################################################
# Visualization of the Depth
#######################################################
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(rows, cols, Z_estimated, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()