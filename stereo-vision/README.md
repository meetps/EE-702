## EE 702 Project Codes

### Bino and Multi-occular Stereo Vision 

#### Group members:
	- Meet Shah (13D070003)
	- Yash Bhalgat (13D070014)

#### Abstract 

This code repository aims at implementing stereo vision using patch-based similarity measures to estimate the disparity and hence generate the depth maps. The patch similarity measures used include 
	* Sum of absolute differences `SAD`
	* Sum of squares of differences `SSD`
	* Normalized Cross Correlation `NCC`

Multi-occular stereo is also implemented using 4 images of the same object with linearly increasing distances between the 1st image. 
	
#### Code files:
	- `stereo_SSD.m` to implement binoccular stereo using SSD.
	- `stereo_NCC.m` to implement binoccular stereo using NCC.
	- `stereo_SAD.m` to implement binoccular stereo using SSD.
	- `dataScript.m` to run all stero methods over the Middlebury dataset.
	- `multiOccular.m` to run multioccular stereo over the Middlebury dataset.
	
#### Result files:
	- The result files for each image in the dataset has been saved in the individual folders with names : 
		* depthSAD.png
		* depthNCC.png
		* depthSSD.png
		* depthMultiOccular.png

#### Observations:

##### SSD 
	* Since the SSD patch matching uses sum of squares of differences, it becomes almost 
	  impossible to distuinguish between patches that have almost the same texture besides having very little differences in depth. This patches lead to convolved zero-boundary features on the texture plane as observed in many depth_SSD files having planes with similar textures.	

##### SAD
	* TBA 
##### NCC
	* TBA 
##### Multi-occular
	* TBA
 
 
