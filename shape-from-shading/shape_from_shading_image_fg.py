import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def changeParams(param1, param2, sourceParamType):
    if sourceParamType == 'pq':
        p = np.array(param1,copy=True)
        q = np.array(param2,copy=True) 
        if param1.shape[0] > 1:
            tanPQ = (p**2 + q**2)**(0.5)
            # plt.imshow(tanPQ)
            # print np.amax(param1)
            # tanFG = np.tan(0.5*np.arctan(tanPQ))
            f = 2*p/(1 + (1+ p**2 + q**2)**(0.5) )
            g = 2*q/(1 + (1+ p**2 + q**2)**(0.5) )
            return f,g
        else:
            tanPQ = (p**2 + q**2)^0.5
            tanFG = np.tan(0.5*np.arctan(tanPQ));
            f = 2*p*tanFG/tanPQ
            g = 2*q*tanFG/tanPQ
            return f,g
    else: 
    #Converting fg => pq
        f = np.array(param1,copy=True)
        g = np.array(param2,copy=True)    
        threshold = 1.5
        if param1.shape[0] > 1:
            denImage = 4-f**2-g**2
            p = 4*f/denImage
            q = 4*g/denImage
            return p, q
        else:
            p = 4*f/(4-f**2-g**2)
            q = 4*g/(4-f**2-g**2)
            return p, q

#######################################################
# Parameter Definition
#######################################################
source = [0,0,1]            # Coordinate of Light Source
Lambda = 100                # Regularization Parameter
noiseSNR = 5;               # Noise to signal ratio
radiusToImageRatio = 0.25   # Radius to Image dimensions ratio
sphereImageSize = 75   # Radius of the spehere to be rendered

#######################################################
# Rendering the 3D Surface
#######################################################
depthMap         = np.zeros((sphereImageSize,sphereImageSize))                     # Array to store depth (z) values
clippingMap         = np.zeros((sphereImageSize,sphereImageSize))                     # Array to store depth (z) values
regionOfInterest = np.zeros((sphereImageSize,sphereImageSize))                     # Boolean flag to mark the sphere ROI
radius           = radiusToImageRatio * sphereImageSize                            # Radius of the sphere
clippingRadius   = radius * 0.95
[cols,rows]      = np.meshgrid(range(0,sphereImageSize),range(0,sphereImageSize))  # Meshgrid for base of computation

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

# print(depthMap)
# print(regionOfInterest)

#######################################################
# Calculating the x, y Gradient Fields p and q
#######################################################
p,q = np.zeros((sphereImageSize,sphereImageSize)),np.zeros((sphereImageSize,sphereImageSize))
for i in range(1,sphereImageSize-1):
    for j in range(1,sphereImageSize-1):
        p[i][j] = depthMap[i][j] - depthMap[i][j-1]
        q[i][j] = depthMap[i][j] - depthMap[i-1][j]  

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
p,q = p * intersectionROI , q * intersectionROI


# print(sum(sum(boundaryMap)))


#######################################################
# Occluding Boundary gradients
#######################################################
# gradX, gradY       = radiance, radiance
gradX, gradY       = np.array(radiance,copy=True),np.array(radiance,copy=True)
#gradX, gradY = np.zeros(depthMap.shape),np.zeros(depthMap.shape)
gradX[:][1:-1]     = (  gradX[:][2:]   - gradX[:][:-2]) * 0.5 
gradY[1:-1][:]     = (  gradY[:-2][:]  - gradY[2:][:] ) * 0.5
# plt.imshow(gradX)
# gradX = gradX * boundaryMap.astype(bool)
# gradY = gradY * boundaryMap.astype(bool)

f,g = changeParams(gradX,gradY,'pq')
# f,g = gradX, gradY

#f,g = f * intersectionROI , g * intersectionROI

pBoundary = np.array(f * boundaryMap.astype(bool),copy=True)
qBoundary = np.array(g * boundaryMap.astype(bool),copy=True)
pBoundary = pBoundary - pBoundary * (1 - boundaryMap)
qBoundary = qBoundary - qBoundary * (1 - boundaryMap)
# print(pBoundary)

#######################################################
# Iterative Shape from shading
#######################################################
limit = 1000
p_next,q_next = np.array(pBoundary,copy=True),np.array(qBoundary,copy=True)
p_estimated,q_estimated = np.array(pBoundary,copy=True),np.array(qBoundary,copy=True)   

for iteration in range(0,limit):
    # print('Starting Iteration :', iteration+1)
    for i in range(1,pBoundary.shape[0] -1):
        for j in range(1,pBoundary.shape[1] -1):
            if regionOfInterestRadiance[i][j] == 1 :
                RadianceX,RadianceY = 0,0
                Rnum = 16*(source[0]*f[i][j] + source[1]*g[i][j]) + (4-f[i][j]**2-g[i][j]**2)*(4-source[0]**2-source[1]**2);
                Rden = (4 + f[i][j]**2 + g[i][j]**2)*(source[0]**2 + source[1]**2 + 4);
                radianc = Rnum / Rden;
                RadianceX = (-1*(16*source[0] - 2*f[i][j]*(4-source[0]**2-source[1]**2)*Rden) - Rnum*(2*f[i][j]*(4+source[0]**2+source[1]**2)))/Rden**2;
                RadianceY = (-1*(16*source[1] - 2*g[i][j]*(4-source[0]**2-source[1]**2)*Rden) - Rnum*(2*g[i][j]*(4+source[0]**2+source[1]**2)))/Rden**2;
                # radianc = (p_estimated[i][j] * source[0] + q_estimated[i][j] * source[1] + 1) / ( ((source[0]**2+source[1]**2 + 1)**0.5) *((p_estimated[i][j]**2 + q_estimated[i][j]**2 + 1)**0.5))
                # RadianceX = (p_estimated[i][j]**2 *source[0] + source[0] - q_estimated[i][j]*q_estimated[i][j]*source[1] - p_estimated[i][j])/(((source[0]**2 + source[1]**2 + 1)**0.5)*(p_estimated[i][j]**2 + (q_estimated[i][j]**2 + 1)**0.5)**3)
                # RadianceY = (q_estimated[i][j]**2 *source[1] + source[1] - p_estimated[i][j]*p_estimated[i][j]*source[0] - q_estimated[i][j])/(((source[0]**2 + source[1]**2 + 1)**0.5)*(p_estimated[i][j]**2 + (q_estimated[i][j]**2 + 1)**0.5)**3)
                p_next[i][j] = 0.25*(p_estimated[i-1][j] + p_estimated[i+1][j] + p_estimated[i][j-1] + p_estimated[i][j+1]) - 1/Lambda *(radiance[i][j] - radianc)*RadianceX;
                q_next[i][j] = 0.25*(q_estimated[i-1][j] + q_estimated[i+1][j] + q_estimated[i][j-1] + q_estimated[i][j+1]) - 1/Lambda *(radiance[i][j] - radianc)*RadianceY;

    p_estimated = (p_next*regionOfInterestRadiance*(1-boundaryMap.astype(bool))) + pBoundary*boundaryMap*regionOfInterestRadiance
    q_estimated = (q_next*regionOfInterestRadiance*(1-boundaryMap.astype(bool))) + qBoundary*boundaryMap*regionOfInterestRadiance

p_est, q_est = changeParams(p_estimated, q_estimated, 'fg')
p_estimated = p_est# * (p_est<2)
q_estimated = q_est# * (q_est<2)

# print(p_estimated[sphereImageSize/2])

#######################################################
# Depth Retrieval 
#######################################################
limit = 1000
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
#Z_estimated = Z_estimated

print(np.amax(radiance))
print(np.amin(radiance))

# print(q_y[sphereImageSize/2])
# print(p_x[sphereImageSize/2])
# print(p_x[sphereImageSize/2])
#######################################################
# Visualization of the Depth
#######################################################
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlim3d(0,sphereImageSize)
ax.set_ylim3d(0,sphereImageSize)
ax.set_zlim3d(0,sphereImageSize)
surf = ax.plot_surface(rows, cols, Z_estimated, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=1, aspect=5)
plt.show()