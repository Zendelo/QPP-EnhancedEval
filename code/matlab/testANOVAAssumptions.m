function []=testANOVAAssumptions(TAG, dataPath)
    common_params;

    dataTable = readtable(dataPath, "delimiter", ",", "format", "%s%s%f%s%s%s%s%s%s%s");
    
    FILTERS = struct();

    dataTable = filterMeasure(dataTable, FILTERS);
    
    [tbl, stats, soa] = computeANOVA(TAG, dataTable);
    
    x = (stats.resid - mean(stats.resid))/std(stats.resid);
    %histogram(x, 'Normalization', 'pdf');
    %hold on
    %histogram(normrnd(0, 1, 1000), 'Normalization', 'pdf');
    qqplot(x);
    [h, p] = kstest(x);
    fprintf("kolmogorov-smirnov: hp: %d - pval: %f\n", h, p);
    cdfplot(x)
    hold on
    x_values = linspace(min(x),max(x));
    plot(x_values,normcdf(x_values,0,1),'r-')
    
    legend('Empirical CDF','Standard Normal CDF','Location','best')
    
end