function X = getSAD(A,B)
	X = sum(sum(abs(A - B)));
end