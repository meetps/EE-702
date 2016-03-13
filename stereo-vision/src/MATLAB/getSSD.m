function X = getSSD(A,B)
	X = sum(sum((A - B).^2));
end