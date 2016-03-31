v0 = imread('../../data/Aloe/view2.png');
v1 = imread('../../data/Aloe/view3.png');
v2 = imread('../../data/Aloe/view4.png');
v3 = imread('../../data/Aloe/view5.png');

d1 = stereo_SAD(v2, v3, 16);
d2 = stereo_SAD(v1, v3, 2*16);
d3 = stereo_SAD(v0, v3, 3*16);

depth = (d1/1 + d2/2 + d3/3 )/3;

depth(:,372:end) = d1(:,372:end);

imshow(depth,[]);
colormap(hot);