%% Calculate the score of result with the ground 
%% truth from the dataset. 

close all;
clear all;
data_path = '../../data/';
directory_list = dir('../../data/');

ground_Truth = 'disp1.png';
depth_NCC    = 'depthNCC.png';
depth_SAD    = 'depthSAD.png';
depth_SSD    = 'depthSSD.png';

score = zeros(size(directory_list,1)-2,4,3);

for i=3:size(directory_list),
	
	folder_path = strcat([data_path directory_list(i).name '/']);
	
	groundTruth = imread(strcat([folder_path ground_Truth ])); 
	depthNCC    = imread(strcat([folder_path depth_NCC ])); 
	depthSSD    = imread(strcat([folder_path depth_SSD ])); 
	depthSAD    = imread(strcat([folder_path depth_SAD ])); 


	score(i,1,:) = [ssim(depthNCC,groundTruth), ssim(depthSAD,groundTruth), ssim(depthSSD,groundTruth)];
	score(i,2,:) = [getSAD(groundTruth,depthNCC), getSAD(groundTruth,depthSAD), getSAD(groundTruth,depthSSD)];
	score(i,3,:) = [getSSD(groundTruth,depthNCC), getSSD(groundTruth,depthSAD), getSSD(groundTruth,depthSSD)];
	score(i,4,:) = [immse(depthNCC,groundTruth), immse(depthSAD,groundTruth), immse(depthSSD,groundTruth)];

	% Add norm correlation as a measure maybe
	% score(i,1,:) = [normxcorr2(groundTruth,depthNCC), normxcorr2(groundTruth,depthSAD), normxcorr2(groundTruth,depthSSD)];
	
	save('result.mat','score');

end