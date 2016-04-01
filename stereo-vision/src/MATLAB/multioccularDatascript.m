%% Test Script to run multiOccular stereo over all images

close all;
clear all;
data_path = '../../data/';
directory_list = dir('../../data/');

maxOffset = 16;

for i=4:size(directory_list),
	folder_path = strcat([data_path directory_list(i).name '/'])
	v0 = imread(strcat([folder_path 'view2.png' ])); 
	v1 = imread(strcat([folder_path 'view3.png' ]));
 	v2 = imread(strcat([folder_path 'view4.png' ]));
        v3 = imread(strcat([folder_path 'view5.png' ]));

	depth_SSD = multiOccular(v0,v1,v2,v3);
 	depth_SSD = stereo_SSD(leftImage, rightImage, maxOffset);
 	depth_NCC = stereo_NCC(leftImage, rightImage, maxOffset);
	
 	imwrite(mat2gray(depth_NCC),strcat([folder_path 'depthNCC.png' ]));
 	imwrite(mat2gray(depth_SAD),strcat([folder_path 'depthSAD.png' ]));
   	figure(1);
    	imshow(mat2gray(depth_SSD), []);
    	saveas(1, strcat([folder_path 'depthMultiOccularSSD_gray' ]), 'png');
	imwrite(mat2gray(depth_SSD),strcat([folder_path 'depthMultiOccularSSD_gray.png' ]));
	close all;
end
