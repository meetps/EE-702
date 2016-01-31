import numpy as np
import cv2
import math

#######################################################
# Parameter Definition
#######################################################
source = [0,0] 				# Coordinate of Light Source
Lambda = 1000 				# Regularization Parameter
noiseSNR = 5; 				# Noise to signal ratio
radiusToImageRatio = 0.1	# Radius to Image dimensions ratio
sphereImageSize = 100	# Radius of the spehere to be rendered

#######################################################
# Rendering the 3D Surface
#######################################################
depthMap         = np.zeros((sphereImageSize,sphereImageSize))
regionOfInterest = np.zeros((sphereImageSize,sphereImageSize)) 
radius           = radiusToImageRatio * sphereImageSize
[cols,rows] 	 = np.meshgrid(range(0,sphereImageSize),range(0,sphereImageSize))

for i in range(0,sphereImageSize):
	for j in range(0,sphereImageSize):
		depthMap[i][j] = math.pow(radius,2) - math.pow(cols[i][j] - sphereImageSize/2 , 2) - math.pow(rows[i][j] - sphereImageSize/2 , 2);
		if(depthMap[i][j] > 0):
			regionOfInterest[i][j] = 1;

depthMap = np.sqrt(depthMap * regionOfInterest)
print(depthMap)
