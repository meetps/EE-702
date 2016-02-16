import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import *

#######################################################
# Parameter Definition
#######################################################
source = [0,0,1] 			 # Coordinate of Light Source
Lambda = 100				 # Regularization Parameter
noiseRadiance = 0.00 		 # Noise to radiance ratio
noiseSource = 0.01           # Noise to source ratio
radiusToImageRatio = 0.25	 # Radius to Image dimensions ratio
sphereImageSize = 50         # Radius of the spehere to be rendered
soslimit = 1000				 # No of iters for Shape from Shading
depthlimit = 1000			 # No of iters for depth retrieval

#######################################################
# Rendering the 3D Surface
#######################################################
print('=====> Starting Sphere Rendering')
depthMap         = np.zeros((sphereImageSize,sphereImageSize))                     # Array to store depth (z) values
clippingMap      = np.zeros((sphereImageSize,sphereImageSize))                     # Array to store depth (z) values
regionOfInterest = np.zeros((sphereImageSize,sphereImageSize))                     # Boolean flag to mark the sphere ROI
radius           = radiusToImageRatio * sphereImageSize                            # Radius of the sphere
clippingRadius   = radius * 0.98
[cols,rows] 	 = np.meshgrid(range(0,sphereImageSize),range(0,sphereImageSize))  # Meshgrid for base of computation

#Calculating the depth using z^2 = r^2 - x^2 - y^2  for each point in the depth map
for i in range(0,sphereImageSize):
	for j in range(0,sphereImageSize):
		depthMap[i][j] = radius**2 - math.pow(cols[i][j] - sphereImageSize/2 , 2) - math.pow(rows[i][j] - sphereImageSize/2 , 2);
		if(depthMap[i][j] > 0):
			regionOfInterest[i][j] = 1;

depthMap = np.sqrt(depthMap * regionOfInterest)

for i in range(0,sphereImageSize):
	for j in range(0,sphereImageSize):
		clippingMap[i][j] = clippingRadius**2 - math.pow(cols[i][j] - sphereImageSize/2 , 2) - math.pow(rows[i][j] - sphereImageSize/2 , 2);
		if(clippingMap[i][j] > 0):
			regionOfInterest[i][j] = 1;

depthMap = depthMap * regionOfInterest
print('=====> Finished Sphere Rendering')

print('=====> Starting Gradient Field Calculation')
#######################################################
# Calculating the x, y Gradient Fields p and q
#######################################################
p,q = np.zeros((sphereImageSize,sphereImageSize)),np.zeros((sphereImageSize,sphereImageSize))
for i in range(1,sphereImageSize-1):
	for j in range(1,sphereImageSize-1):
		p[i][j] = depthMap[i][j] - depthMap[i][j-1]
		q[i][j] = depthMap[i][j] - depthMap[i-1][j]  

p,q = p * regionOfInterest, q * regionOfInterest

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


print('=====> Finished Gradient Field Calculation')

print('=====> Starting Boundary Gradient Calculation')
#######################################################
# Detecting Boundary using Morphological Operations from OpenCV
#######################################################

boundaryMap              = np.zeros((sphereImageSize,sphereImageSize))
regionOfInterestRadiance = radiance > 0
kernel                   = np.ones((3,3),np.uint8)
boundaryMap              = regionOfInterestRadiance - cv2.erode(regionOfInterestRadiance.astype(np.uint8),kernel,5)
boundaryMap              = cv2.erode(cv2.dilate(boundaryMap.astype(np.uint8),kernel,3),kernel,3) 
intersectionROI          = regionOfInterest * regionOfInterestRadiance
p,q = p * intersectionROI , q * intersectionROI

noiseR = np.random.normal(0,1,sphereImageSize*sphereImageSize)
noiseR = noiseR.reshape(sphereImageSize,sphereImageSize)*noiseRadiance
radiance = radiance + noiseR
source = source + np.random.normal(0,1,3)*noiseSource

#######################################################
# Occluding Boundary gradients
#######################################################
gradX, gradY       = np.array(radiance,copy=True),np.array(radiance,copy=True)
gradX[:][1:-1]     = (  gradX[:][2:]   - gradX[:][:-2]) * 0.5 
gradY[1:-1][:]     = (  gradY[:-2][:]  - gradY[2:][:] ) * 0.5

pBoundary = np.array(gradX * boundaryMap.astype(bool),copy=True)
qBoundary = np.array(gradY * boundaryMap.astype(bool),copy=True)
pBoundary = pBoundary - pBoundary * (1 - boundaryMap)
qBoundary = qBoundary - qBoundary * (1 - boundaryMap)
print('=====> Finished Boundary Gradient Calculation')

print('=====> Starting Iterative Shape from shading')
#######################################################
# Iterative Shape from shading
#######################################################
p_next,q_next = np.array(pBoundary,copy=True),np.array(qBoundary,copy=True)
p_estimated,q_estimated = np.array(pBoundary,copy=True),np.array(qBoundary,copy=True)	

for iteration in tqdm(range(0,soslimit)):
	# print('Starting Iteration :', iteration+1)
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
print('=====> Finished Iterative Shape from shading')

print('=====> Starting Depth Retrieval')
#######################################################
# Depth Retrieval 
#######################################################
Z_p   = np.zeros(p_estimated.shape)
Z     = np.zeros(p_estimated.shape)
p_x,q_y = np.array(p_estimated,copy=True),np.array(q_estimated,copy=True)
p_x[:][1:-1] = ( p_estimated[:][2:] - p_estimated[:][:-2]);
q_y[1:-1][:] = ( q_estimated[:-2][:] - q_estimated[2:][:]);
for iteration in tqdm(range(0,depthlimit)):
	for i in range(1,p_estimated.shape[0]-1):
		for j in range(1,p_estimated.shape[1]-1):
			if regionOfInterestRadiance[i][j] == 1 :
				Z[i][j] = 0.25*( Z_p[i-1][j] + Z_p[i+1][j] + Z_p[i][j-1] + Z_p[i][j+1]) + abs(p_x[i][j]) + abs(q_y[i][j])
	Z_p = regionOfInterestRadiance*Z;

Z_estimated = Z * regionOfInterestRadiance
# Z_estimated = Z_estimated

# print(np.amax(radiance))
# print(np.amin(radiance))

print('=====> Finished Depth Retrieval')
#######################################################
# Visualization of the Depth
#######################################################
plt.imshow(Z_estimated)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d(0,sphereImageSize)
ax.set_ylim3d(0,sphereImageSize)
ax.set_zlim3d(0,sphereImageSize)
filename = 'r_' + str(sphereImageSize) + 'nr_' + str(noiseRadiance) + 'ns_' + str(noiseSource) + 'lambda_' + str(Lambda) + '_pq'
np.save('results/' + filename ,Z_estimated)
surf = ax.plot_surface(rows, cols, Z_estimated, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=1, aspect=5)
plt.show()
