function [normCrossCorr] = getNCC(rightImage,leftImage,i,j,k,windowRange)
	squareMeanR = rightImage(i+windowRange,j+windowRange) / sqrt(sum(sum((rightImage(i+windowRange,j+windowRange) .* rightImage(i+windowRange,j+windowRange)))));
	squareMeanL = leftImage(i+windowRange,k+windowRange) / sqrt(sum(sum((leftImage(i+windowRange,k+windowRange).* leftImage(i+windowRange,k+windowRange)))));
	normCrossCorr = corr2(squareMeanR,squareMeanL);
end