function []=simpleANOVAnoHardTopics(TAG, dataPath)
    common_params;

    dataTable = readtable(dataPath, "delimiter", ",", "format", "%s%s%f%s%s%s%s%s%s%s");
    
    deselectedTopics = {'311', '693', '320', '681', '625', '679', '356'};
    
    topics = unique(dataTable{:, 'topic'});
    
    selectedTopics = topics(~ismember(topics, deselectedTopics));
    
    FILTERS = struct();
    %FILTERS.retrievalFunction = ["PRE"];
    FILTERS.topic = selectedTopics;
    dataTable = filterMeasure(dataTable, FILTERS);
    
    
    [tbl, stats, soa] = computeANOVA(TAG, dataTable);
    disp(getLatexANOVATable(TAG, tbl, soa));
    
    %disp(multcompare(stats));


end