## EE 702 Project Codes

### Multi and Binocular Stereo Vision using SAD, SSD and NCC 

#### Group members:
	- Meet Shah (13D070003)
	- Yash Bhalgat (13D070014)
	
#### Code files:
	- stereo_SSD.py to implement binocular stereo using SSD.
	
#### Result files:

#### Observations:

##### SSD 
	* Since the SSD patch matching uses sum of squares of differences, it becomes almost 
	  impossible to distuinguish between patches that have almost the same texture besides having very little differences in depth. This patches lead to convolved zero-boundary features on the texture plane as observed in many depth_SSD files having planes with similar textures.	

##### SAD 
##### NCC 
