function []=plotANOVAMainEffects(TAG, dataPath)
    common_params;
    
    dataTable = readtable(dataPath, "delimiter", ",", "format", "%s%s%f%s%s%s%s%s%s%s");
    
    for rn = 1:length(EXPERIMENT.analysis.(TAG).model)
        idxs = EXPERIMENT.analysis.(TAG).model(rn, :);
        if sum(idxs)==1
            label = EXPERIMENT.analysis.(TAG).labels(find(idxs));
            factors = [{'sARE'}, label(:)'];
            means = grpstats(dataTable(:, factors), label);
            cis = grpstats(dataTable(:, factors), label, @(x) confidenceIntervalDelta(x, 0.05));
            order = means{:, label{1}};
            xtl = order;

            means = sortrows(means(:, ["mean_sARE", label{1}]), "mean_sARE", 'descend');
            order = means{:, label{1}};
            xtl = order;
            for xtln=1:length(xtl)
                xtl{xtln} = replace(xtl{xtln}, "-idf", "IDF");
                xtl{xtln} = replace(xtl{xtln}, "scq", "SCQ");
                xtl{xtln} = replace(xtl{xtln}, "-SCQ", "SCQ");
                xtl{xtln} = replace(xtl{xtln}, "var", "VAR");
                xtl{xtln} = replace(xtl{xtln}, "-VAR", "VAR");
                xtl{xtln} = replace(xtl{xtln}, "wig", "WIG");
                xtl{xtln} = replace(xtl{xtln}, "-WIG", "(WIG)");
                xtl{xtln} = replace(xtl{xtln}, "nqc", "NQC");
                xtl{xtln} = replace(xtl{xtln}, "-NQC", "(NQC)");
                xtl{xtln} = replace(xtl{xtln}, "smv", "SMV");
                xtl{xtln} = replace(xtl{xtln}, "-SMV", "(SMV)");
                xtl{xtln} = replace(xtl{xtln}, "clarity", "Clarity");
                xtl{xtln} = replace(xtl{xtln}, "-Clarity", "(Clarity)");
                xtl{xtln} = replace(xtl{xtln}, "uef", "UEF");
            end
        
            currentFigure = figure('Visible', 'on');
            
            hold on;
            ax = gca;
            ax.TickLabelInterpreter = 'latex';
            ax.FontSize = 32;

           
            if ~strcmp(label, 'todpic')
                
                h = errorbar(1:height(means), means{order, 'mean_sARE'}, cis{order, 'Fun1_sARE'}*1.75,...
                'Marker', 'o',...
                'MarkerSize',10);
            else
                h = plot(1:height(means), means{order, 'mean_sARE'},...
                    'Marker', 'o',...
                    'MarkerSize',10);
            end
            set(h, 'MarkerEdgeColor', get(h,'Color'));
            set(h, 'Linewidth', 2);
            set(h, 'MarkerFaceColor', 'w');


            if height(means)<20
                ax.XTick = 1:(height(means));
                ax.XTickLabel = xtl;
            end
            ax.XTickLabelRotation = 45;

            ax.XLabel.String = label{1};
            ax.XLabel.Interpreter = 'latex';
            ax.YLabel.String = 'mean sARE';
            ax.YLabel.Interpreter = 'latex';

            ax.XGrid = 'off';
            ax.YGrid = 'on';

            currentFigure.PaperPositionMode = 'auto';
            currentFigure.PaperUnits = 'centimeters';
            currentFigure.PaperSize = [126 44];
            currentFigure.PaperPosition = [1 1 120 40];
            
            dataName = split(dataPath, "/");
            dataName = dataName(end);
            dataName = split(dataName, ".");
            dataName = dataName(1);
      
            fname = sprintf("../../data/plots/%s_%s_%s.pdf",TAG, dataName, label{1});
            print(currentFigure, '-dpdf', fname);
            close(currentFigure);
        end
    end
    
    
    
end


function [delta] = confidenceIntervalDelta(data, alpha, dim)
    
    narginchk(2, 3);
    
    if nargin < 3
        dim = 1;
    end

    n = size(data, dim);

    delta = tinv(1 - alpha/2, n - 1) * std(data, 0, dim) / sqrt(n);
end