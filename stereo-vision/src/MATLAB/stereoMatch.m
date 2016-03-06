function  [disparityMap, disparityMask ] = stereoMatch(rightImage,leftImage, edgeRight, edgeLeft, corrWindowSize, minOffset, maxOffset, matchType)
	[ height,width ] = size(rightImage);
	disparityMap = zeros(height,width);
	disparityMask = zeros(height,width);
	windowSize = (corrWindowSize-1)/2;
	windowRange  = -windowSize:windowSize;
	h = waitbar(0,'Computing Depth');
	for i = 1 + windowSize:height - windowSize,
		for j = 1 + windowSize:width - windowSize,
			if(edgeRight(i,j))
				maxCorr = getNCC(rightImage,leftImage,i,j,windowRange);
				for k=j+1:width-windowSize-1,
					Corr = getNCC(rightImage,leftImage,i,k,windowRange);
					if(maxCorr < Corr)
						maxCorr = Corr;
						disparityMap(i,j) = k-j;
					end
				end
				disparityMask(i,j)	= 1;
				%if( or disparityMap(i,j) > maxOffset):
				if(maxCorr < 0.7 || disparityMap(i,j) > maxOffset)
					disparityMap(i,j) = 0;
					edgeRight(i,j) = 0;
					disparityMask(i,j) = 0;
				end
			else
				disparityMask(i,j)	= 0;
			end
		end
	%disp(num2str(i))	
	waitbar(i/ height-windowSize);
	end
	close(h)
end	