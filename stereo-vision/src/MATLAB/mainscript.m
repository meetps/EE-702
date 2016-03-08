%% Mainscript

close all;
clear all;

scale = 1;

leftImage = double(rgb2gray(imresize(imread('view0.png'),scale)));
rightImage = double(rgb2gray(imresize(imread('view1.png'),scale)));

edgeRight = edge(rightImage, 'canny',0.01);
edgeLeft = edge(leftImage, 'canny',0.01);

maxDisparity = round(60 * scale);
maxOffset = round(60 * scale);
minDisparity = 0;
minOffset = 0;
corrWindowSize = 2.*round((25 * scale + 1)/2) +1;

[ depthMap, disparityMask ] = stereoMatch(rightImage,leftImage, edgeRight, edgeLeft, corrWindowSize, minDisparity, maxDisparity, 'NCC'); 

figure;
imshow(depthMap,[])
finalDepth  = postProcessDepth(depthMap,2000,edgeRight);
figure;
imshow(finalDepth,[])