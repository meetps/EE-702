function disparityMap = stereo_NCC(leftImage, rightImage, maxDisparity)

    if(size(leftImage,3)==3)
        leftImage = rgb2gray(leftImage);
    end
    if(size(rightImage,3)==3)
        rightImage = rgb2gray(rightImage);
    end
    
    [height, width] = size(leftImage);
    
    leftImage = double(leftImage);
    rightImage = double(rightImage);
    
    winRangedowSize = 15;
    winRange = (winRangedowSize-1)/2;

    minDisparity = 0.0;

    disparityMap = zeros(height, width);
    
    for i = 1+winRange:height-winRange,
        for j = 1+winRange:width-winRange-maxDisparity,
            maxCorr = 0.0;
            offset = minDisparity;
            for dispRange = minDisparity:maxDisparity,
                Correl = 0.0;
                t = -winRange:winRange;

                dotLeftRight = sum(sum(rightImage(i+t,j+t).*leftImage(i+t,j+t+dispRange)));
                squaredSumRight = sum(sum(rightImage(i+t,j+t).*rightImage(i+t,j+t)));
                squaredSumLeft = sum(sum(leftImage(i+t,j+t+dispRange).*leftImage(i+t,j+t+dispRange)));

                Correl=dotLeftRight/sqrt(squaredSumLeft*squaredSumRight);

                if (maxCorr < Correl)
                    maxCorr = Correl;
                    offset = dispRange;
                end
            end
            disparityMap(i,j) = offset;
        end
    end
end