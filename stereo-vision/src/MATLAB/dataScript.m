%% Test Script to run stereo over all images

close all;
clear all;
data_path = '../../data/';
directory_list = dir('../../data/');

rightView  = 'view1.png';
leftView  = 'view0.png';

for i=3:size(directory_list),
	folder_path = strcat([data_path directory_list(i).name '/'])
	rightImage = imread(strcat([folder_path rightView ])); 
	leftImage = imread(strcat([folder_path leftView ]));

	depth_SAD = stereo_SAD(leftImage, rightImage, maxOffset);
	depth_SSD = stereo_SSD(leftImage, rightImage, maxOffset);
	depth_NCC = stereo_NCC(leftImage, rightImage, maxOffset);
	
	imwrite(mat2gray(depth_NCC),strcat([folder_path 'depthNCC.png' ]));
	imwrite(mat2gray(depth_SAD),strcat([folder_path 'depthSAD.png' ]));
	imwrite(mat2gray(depth_SSD),strcat([folder_path 'depthSSD.png' ]));
end