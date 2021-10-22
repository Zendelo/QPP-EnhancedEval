function []=ANOVAOnBinnedDifficulty(TAG, vtype, nbins)
    
    distributional_data_dir = "../../data/distributional_measures/";
    collection = "RQV04";
    stdMeasure = "map";
    tieHandling = "average";
    qppMeasure = "sARE";
    cutStrategy = "qcut";
    
    common_params;
    fpath = "%sDistributionalMeasure_%s_%s_%s_%s_%s_%s_%d_%d.csv";
    for i=0:(nbins-1)
        
        %name of the file to be loaded
        fname = sprintf(fpath, distributional_data_dir, collection, ...
                        stdMeasure, qppMeasure, tieHandling, vtype, ...
                        cutStrategy, i, nbins);
                    
                    
        dataTable = readtable(fname, "delimiter", ",", "format", "%s%s%f%s%s%s%s%s%s");

                    
                    
        [tbl, ~, soa] = computeANOVA(TAG, dataTable);
        disp(getLatexANOVATable(TAG, tbl, soa));
    end

end