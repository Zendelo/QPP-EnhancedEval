function factorLabels = getFactorLabels(TAG)

    common_params;
    
    factorNames = EXPERIMENT.analysis.(TAG).labels;
    
    md = EXPERIMENT.analysis.(TAG).model;
    
    if any(strcmp(fieldnames(EXPERIMENT.analysis.(TAG)), 'nested'))
        [r, c] = find(EXPERIMENT.analysis.(TAG).nested);
        for idx=1:length(r)
            fn = [factorNames{r(idx)}, '_l_', factorNames{c(idx)}, '_r_'];
            factorNames{r(idx)} = fn;
        end
    end
    
    [nrows, ~] = size(md);
    
    factorLabels = cell(sum(any(md, 1)), 1);
    currLabel = 1;
    for rn = 1:nrows
        idx = boolean(md(rn, :));
        if any(idx)
            fn = char(join(factorNames(idx), "_"));
            factorLabels{currLabel} = fn;
            currLabel = currLabel + 1;
        end
    end
    
   

end