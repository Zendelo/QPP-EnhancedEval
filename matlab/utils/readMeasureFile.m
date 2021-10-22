function [tbl, varargout] = readMeasureFile(collection)
    
    tbl = readtable(sprintf("../../data/measures/%s_mfile.csv", collection), "delimiter", ",");
    tbl.Properties.VariableNames = ["model", "class", "topic", "measure"];
    

    nout = max(nargout,1) - 1;
    if nout>=1
         varargout{1} = tbl(strcmp(tbl{:, "class"}, "fus"), :);
    end
    
    tbl = tbl(~strcmp(tbl{:, "class"}, "fus"), :);


end

