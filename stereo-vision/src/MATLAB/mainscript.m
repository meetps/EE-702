%% Mainscript
close all;
clear all;

leftImage = double(rgb2gray(imread('view0.png')));
rightImage = double(rgb2gray(imread('view1.png')));

edgeRight = edge(rightImage, 'canny',0.01);
edgeLeft = edge(leftImage, 'canny',0.01);

maxDisparity = 60;
minDisparity = 0;
corrWindowSize = 25;

[ depthMap, disparityMask ] = stereoMatch(rightImage,leftImage, edgeRight, edgeLeft, corrWindowSize, minDisparity, maxDisparity, 'NCC'); 
imshow(depthMap,[])
finalDepth  = postprocessDepth(depthMap,2000,edgeRight);
imshow(finalDepth,[])