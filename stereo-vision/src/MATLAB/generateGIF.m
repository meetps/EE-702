%% Script to generate GIF from result images

close all;
clear all;
data_path = '../../data/';
directory_list = dir('../../data/');

% Left GIF
rightView  		= 'view1.png';
leftView  		= 'view0.png';
groundTruth 	= 'disp1.png';

% Right GIF
NCCview  		= 'depthNCC.png';
SADview  		= 'depthSAD.png';
SSDview      	= 'depthSSD.png';
mNCCview		= 'depthMultiOccularNCC.png'; 
mSADview		= 'depthMultiOccularSAD.png';
mSSDview 		= 'depthMultiOccularSSD.png';

maxOffset = 16;

for i=4:size(directory_list)-18,
	folder_path = strcat([data_path directory_list(i).name '/'])
	
	rightImage = imread(strcat([folder_path rightView ])); 
	leftImage = imread(strcat([folder_path leftView ]));
	groundTruthImage = imread(strcat([folder_path groundTruth]));

	NCC  = imread(strcat([folder_path NCCview]));
	SAD  = imread(strcat([folder_path SADview]));
	SSD  = imread(strcat([folder_path SSDview]));
	mNCC = imread(strcat([folder_path mNCCview]));
	mSAD = imread(strcat([folder_path mSADview]));
	mSSD = imread(strcat([folder_path mSSDview]));
  	
  	ImageCell = {rightImage; leftImage};
  	FileName = 'input.gif';
	for k = 1:numel(ImageCell)
    	if k ==1
        	imwrite(ImageCell{k},FileName,'gif','LoopCount',Inf,'DelayTime',1);
    	else
        	imwrite(ImageCell{k},FileName,'gif','WriteMode','append','DelayTime',1);
    	end
	end

  	ImageCell = [NCC , SAD, SSD];
    FileName = 'binoOutput.gif';
	for k = 1:numel(ImageCell)
    	if k ==1
        	imwrite(ImageCell{k},FileName,'gif','LoopCount',Inf,'DelayTime',1);
    	else
        	imwrite(ImageCell{k},FileName,'gif','WriteMode','append','DelayTime',1);
    	end
	end

    ImageCell = [mNCC, mSAD, mSSD];
    FileName = 'multiOutput.gif';	
	for k = 1:numel(ImageCell)
    	if k ==1
        	imwrite(ImageCell{k},FileName,'gif','LoopCount',Inf,'DelayTime',1);
    	else
        	imwrite(ImageCell{k},FileName,'gif','WriteMode','append','DelayTime',1);
    	end
	end
	close all;
end

