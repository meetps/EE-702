function [normCrossCorr] = getNCC(rightImage,leftImage,i,j,windowRange)
	squareMeanR = rightImage(i+windowRange,j+windowRange) / sqrt(sum(sum((rightImage(i+windowRange,j+windowRange)))));
	squareMeanL = leftImage(i+windowRange,j+windowRange) / sqrt(sum(sum((rightImage(i+windowRange,j+windowRange)))));
	normCrossCorr = corr2(squareMeanR,squareMeanL);
end