addpath(genpath('anova_models'));
addpath(genpath('utils'));



EXPERIMENT.analysis.anova.sstype = 3;
EXPERIMENT.analysis.alpha.threshold = 0.05;


EXPERIMENT.analysis.getSelectedFactors = @(labels, FACTORS) getSelectedFactors(labels, FACTORS);


MD0; MDqNestedDiff; MD2qNestedDiff; MD1; MD2; MD3;

function [selectedFactors] = getSelectedFactors(labels, FACTORS)
    selectedFactors = cell(1, length(labels));
    for ln=1:length(labels)
        lb = labels{ln};
        selectedFactors{ln} = FACTORS.(lb);
    end
end