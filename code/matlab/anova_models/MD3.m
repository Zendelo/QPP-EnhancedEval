EXPERIMENT.analysis.MD3.labels = {'query', 'topic', 'stoplist', 'stemmer', 'predictor'};

EXPERIMENT.analysis.MD3.model = [eye(length(EXPERIMENT.analysis.MD3.labels));
                                 0 1 1 0 0;
                                 0 1 0 1 0;
                                 0 1 0 0 1;
                                 1 0 1 0 0;
                                 1 0 0 1 0;
                                 1 0 0 0 1;
                                 0 0 1 1 0;
                                 0 0 1 0 1;
                                 0 0 0 1 1];
                       
EXPERIMENT.analysis.MD3.nested = zeros(length(EXPERIMENT.analysis.MD3.labels));
EXPERIMENT.analysis.MD3.nested(1, 2) = 1;
EXPERIMENT.analysis.MD3.description = "MD3micro ANOVA";


EXPERIMENT.analysis.MD3.compute = @(data, FACTORS)...
  anovan(...
    data, ...
    EXPERIMENT.analysis.getSelectedFactors(EXPERIMENT.analysis.MD3.labels, FACTORS), ... %groups labels
    'model', EXPERIMENT.analysis.MD3.model, ...
    'nested', EXPERIMENT.analysis.MD3.nested, ...
    'VarNames', EXPERIMENT.analysis.MD3.labels, ...
    'sstype', EXPERIMENT.analysis.anova.sstype, ...
    'alpha', EXPERIMENT.analysis.alpha.threshold, ...
    'display', 'off'...
  );



