function []=plotANOVAInteractionEffects(TAG, dataPath, labels)
    common_params;
    dataTable = readtable(dataPath, "delimiter", ",", "format", "%s%s%f%s%s%s%s%s%s%s");
    
        
    FILTERS = struct();
    FILTERS.retrievalFunction = ["QL"];
    
    dataTable = filterMeasure(dataTable, FILTERS);
    

    factors = [{'sARE'}, labels(:)'];
    means = grpstats(dataTable(:, factors), labels);

    xNames = unique(means{:, labels{1}});
    
    if strcmp(labels{1}, 'topic')
        xNames = {xNames{1:find(strcmp(xNames, '672'))-1}, xNames{find(strcmp(xNames, '672'))+1:end}}';
        xNames = randsample(xNames, 50);
        xNames = sort(xNames);
    end  
    yNames = unique(means{:, labels{2}});
    
    
    if strcmp(labels{1}, 'predictor')
        xtl = renamePredictorLabels(xNames);
    else
        xtl = xNames;
    end
    
    if strcmp(labels{2}, 'predictor')
        yl = renamePredictorLabels(yNames);
       
    else
        yl = yNames;
    end
    
    meanMtx = zeros(length(xNames), length(yNames));
    for xn=1:length(xNames)
        xIdx = strcmp(xNames{xn}, means{:, labels{1}});
        for yn=1:length(yNames)
            yIdx = strcmp(yNames{yn}, means{:, labels{2}});
            meanMtx(xn, yn) = means{xIdx & yIdx, 'mean_sARE'};
        end
    end
    
    
    currentFigure = figure('Visible', 'off');

    hold on;
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    ax.FontSize = 32;

    cols = [0.86, 0.37, 0.34;
             0.86, 0.76, 0.34;
             0.57, 0.86, 0.34;
             0.34, 0.86, 0.5;
             0.34, 0.83, 0.86;
             0.34, 0.44, 0.86;
             0.63, 0.34, 0.86;
             0.86, 0.34, 0.70];
    colororder(cols);
    h = plot(1:length(xNames), meanMtx,'Marker', 'o');
    %colormap(ax, parula(10));

    for ln=1:length(h)
        set(h(ln), 'MarkerFaceColor', get(h(ln),'Color')); 
    end
    
    if length(xtl)<20
        ax.XTick = 1:length(xtl);
        ax.XTickLabel = xtl;
        ax.XTickLabelRotation = 45;
    end

    ax.XLabel.String = labels{1};
    ax.XLabel.Interpreter = 'latex';
    ax.YLabel.String = 'mean sARE';
    ax.YLabel.Interpreter = 'latex';
    
    lgd = legend(yl,'Orientation','horizontal', 'Interpreter', 'latex', 'fontsize', 20, 'Location','NorthOutside');
    lgd.NumColumns = 8;

    
    currentFigure.PaperPositionMode = 'auto';
    currentFigure.PaperUnits = 'centimeters';
    currentFigure.PaperSize = [42 22];
    currentFigure.PaperPosition = [1 1 40 20];

    dataName = split(dataPath, "/");
    dataName = dataName(end);
    dataName = split(dataName, ".");
    dataName = dataName(1);

    fname = sprintf("../../data/plots/%s_%s_%s_%s.pdf",TAG, dataName, labels{1}, labels{2});
    print(currentFigure, '-dpdf', fname);
    close(currentFigure);
end

    
    
function [finalStr]=renamePredictorLabels(xNames)

    finalStr = xNames;
    finalStr = replace(finalStr, "-idf", "IDF");
    finalStr = replace(finalStr, "scq", "SCQ");
    finalStr = replace(finalStr, "-SCQ", "SCQ");
    finalStr = replace(finalStr, "var", "VAR");
    finalStr = replace(finalStr, "-VAR", "VAR");
    finalStr = replace(finalStr, "wig", "WIG");
    finalStr = replace(finalStr, "-WIG", "(WIG)");
    finalStr = replace(finalStr, "nqc", "NQC");
    finalStr = replace(finalStr, "-NQC", "(NQC)");
    finalStr = replace(finalStr, "smv", "SMV");
    finalStr = replace(finalStr, "-SMV", "(SMV)");
    finalStr = replace(finalStr, "clarity", "Clarity");
    finalStr = replace(finalStr, "-Clarity", "(Clarity)");
    finalStr = replace(finalStr, "uef", "UEF");
end