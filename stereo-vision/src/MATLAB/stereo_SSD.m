function disparityMap = stereo_SSD(leftImage, rightImage, maxDisparity)
    
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
            minssd = 65535;
            temp=0.0;
            offset = minDisparity;
            for dispRange = minDisparity:maxDisparity,
                ssd=0.0;
                t = -winRange:winRange;
                if (j+winRange+dispRange <= width)
                    ssd=sum(sum((rightImage(i+t,j+t)-leftImage(i+t,j+t+dispRange)).^2));
                end
                if (minssd > ssd)
                    minssd = ssd;
                    offset = dispRange;
                end
            end
            disparityMap(i,j) = offset;
        end
    end
end