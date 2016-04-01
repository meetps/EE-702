function depth = multiOccular(v0,v1,v2,v3)

d1 = stereo_SSD(v2, v3, 16);
d2 = stereo_SSD(v1, v3, 2*16);
d3 = stereo_SSD(v0, v3, 3*16);

depth = (d1/1 + d2/2 + d3/3 )/3;

depth(:,372:end) = d1(:,372:end);

end
