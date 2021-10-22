function []=simpleANOVA(TAG, dataPath)
    common_params;

    dataTable = readtable(dataPath, "delimiter", ",", "format", "%s%s%f%s%s%s%s%s%s%s");
    
    FILTERS = struct();
    %FILTERS.retrievalFunction = ["PRE"];
    
    dataTable = filterMeasure(dataTable, FILTERS);
    
    
    [tbl, stats, soa] = computeANOVA(TAG, dataTable);
    disp(getLatexANOVATable(TAG, tbl, soa));

end