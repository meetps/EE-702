function [filteredDepth] = postProcessDepth(depthMap,iters,edgeRight)
	filteredDepth = zeros(size(depthMap));
	for k=1:iters,
		for i=2:size(depthMap,1)-1,
			for j=2:size(depthMap,2)-1,
				filteredDepth(i,j) = (depthMap(i,j+1) + depthMap(i,j-1) + depthMap(i+1,j) + depthMap(i-1,j))/4;
			end
		end
	depthMap = filteredDepth;
	end
end