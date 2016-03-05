import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import *

RIGHT_IMAGE = 'view0.png'
LEFT_IMAGE = 'view1.png'

def load_image( infilename ) :
	img = cv2.imread(infilename,0)
	data = np.asarray(img, dtype='double')
	return data

def save_image( npdata, outfilename ) :
	img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
	img.save( outfilename )

def getDepthMap(rightImage,leftImage, corrWindowSize , minOffset, maxOffset, matchType ):
	height, width = rightImage.shape[0], rightImage.shape[1]
	disparityMap = np.zeros((height,width),dtype='double')
	windowSize = (corrWindowSize - 1) / 2
	maxFlag = 0
	if matchType in ['NCC','ZNCC']:
		maxFlag = 1

	for i in tqdm(range(windowSize,height - windowSize)):
		for j in range(windowSize, width - windowSize - maxOffset):
			if(maxFlag):
				prevScore = 0.0
			else:
				prevScore = 65532.0 	
			optimiumDisp = minOffset

			for d in range(minOffset,maxOffset):
				regionLeft  = np.array(leftImage[i-windowSize : i+windowSize, j-windowSize : j+windowSize],copy=True)
				regionRight = np.array(rightImage[i-windowSize : i+windowSize, j+d-windowSize : j+d+windowSize],copy=True)
				meanLeft    = np.average(regionLeft.flatten())
				meanRight   = np.average(regionRight.flatten())            
				patchCorrScore = np.zeros((regionLeft.shape))

			if matchType == 'SAD' :
				patchCorrScore = abs(regionLeft - regionRight)
			elif matchType == 'ZSAD':
				patchCorrScore = abs(regionLeft - meanLeft - regionRight + meanRight)
			elif matchType == 'LSAD':
				patchCorrScore = abs(regionLeft - meanLeft/meanRight*regionRight)
			elif matchType == 'SSD':
				patchCorrScore = (regionLeft - regionRight)**2
			elif matchType == 'ZSSD':
				patchCorrScore = (regionLeft - meanLeft - regionRight + meanRight)**2
			elif matchType == 'LSSD':
				patchCorrScore = (regionLeft - meanLeft/meanRight*regionRight)**2
			elif matchType == 'NCC':
				den = (sum(sum(regionLeft**2))*sum(sum(regionRight**2)))**0.5
				patchCorrScore = regionLeft*regionRight/den;
			elif matchType == 'ZNCC':
				# % Calculate the term in the denominator (var: den)
				den = (sum(sum((regionLeft - meanLeft)**2))*sum(sum((regionRight - meanRight)**2)))**0.5
				patchCorrScore = (regionLeft - meanLeft)*(regionRight - meanRight)/den
			# % Compute the final score by summing the values in patchCorrScore,
			# % and store it in a temporary variable signifying the distance
			# % (var: corrScore)
			corrScore=sum(sum(patchCorrScore))
			if(maxFlag):
				if(corrScore>prevScore):
					# % If the current disparity value is greater than
					# % previous one, then swap them
					prevScore=corrScore
					optimiumDisp=d
			else:
				if (prevScore > corrScore):
					# print(corrScore)
					# % If the current disparity value is less than
					# % previous one, then swap them
					prevScore = corrScore
					optimiumDisp = d
			disparityMap[i][j] = optimiumDisp
	return disparityMap

rightImage = load_image(RIGHT_IMAGE)
leftImage = load_image(LEFT_IMAGE)

# print rightImage.shape

depthMap = getDepthMap(rightImage,leftImage,9,0,16,'NCC')
print(np.amax(depthMap))
imgplot = plt.imshow(depthMap,cmap="hot")
plt.show()