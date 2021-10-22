function [tbl, stats, soa] = computeANOVA(TAG, dataTable)

    common_params;    


    for nVar = 1:length(dataTable.Properties.VariableNames)
        f = dataTable.Properties.VariableNames{nVar};
        FACTORS.(f) = dataTable{:, f};
    end
    
    [~, tbl, stats] = EXPERIMENT.analysis.(TAG).compute(FACTORS.sARE, FACTORS);
    
    fl = getFactorLabels(TAG);
    soa = computeSOA(height(dataTable), tbl, fl);
    
    
end