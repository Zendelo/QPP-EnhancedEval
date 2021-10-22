function soa = computeSOA(N, tbl, factors)

	df_error = tbl{end-1, 3};
	ss_error = tbl{end-1, 2};
	F_error  = tbl{end-1, 6};

	for i=1:length(factors)
		dff = tbl{i+1, 3};
		ssf = tbl{i+1, 2};
		Ff  = tbl{i+1, 6};
		soa.omega2p.(factors{i}) = dff * (Ff - 1) / (dff * (Ff - 1) + N);
		soa.eta2p.(factors{i}) = ssf / (ssf + ss_error);
		soa.f2.(factors{i}) = ssf / ss_error;

	end

end