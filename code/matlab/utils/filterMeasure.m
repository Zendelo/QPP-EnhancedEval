function filteredMeasure = filterMeasure(measure, FILTERS)
    
    fieldN = fieldnames(FILTERS);
    selected = ones(height(measure), 1);
    for fn=1:length(fieldN)
        fieldName = fieldN{fn};
        selected = selected .* ismember(measure{:, fieldName}, FILTERS.(fieldName)); 
    end

    
    filteredMeasure = measure(boolean(selected), :);
end