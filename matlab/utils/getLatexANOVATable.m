function [outString] = getLatexANOVATable(TAG, tbl, soa)

    common_params;


    outString = sprintf('\\begin{table}[!ht]\n');
    outString = sprintf('%s\\centering\n', outString);
    outString = sprintf('%s\\mycaption{%s}\n', outString, EXPERIMENT.analysis.(TAG).description);
    outString = sprintf('%s\\label{tab:dtl-%s}\n', outString, TAG);
    outString = sprintf('%s\\resizebox{0.95\\columnwidth}{!}{', outString);
    outString = sprintf('%s\\begin{tabular}{lrrrrrr} \n', outString);

    outString = sprintf('%s\\hline \n', outString);

    outString = sprintf('%s\\multicolumn{1}{c}{\\textbf{Source}} & \\multicolumn{1}{c}{\\textbf{SS}} & \\multicolumn{1}{c}{\\textbf{DF}} & \\multicolumn{1}{c}{\\textbf{MS}} & \\multicolumn{1}{c}{\\textbf{F}} & \\multicolumn{1}{c}{\\textbf{p-value}} & \\multicolumn{1}{c}{$\\hat{\\omega}_{\\langle fact\\rangle}^2$} \\\\ \n', outString);

    outString = sprintf('%s\\hline \n', outString);
    tl = getFactorLabels(TAG);

    for j=1:length(tl)
        lb = tl{j};
        textlb = strrep(lb, "_l_", "(");
        textlb = strrep(textlb, "_r_", ")");
        textlb = strrep(textlb, "_", "*");
        
        outString = sprintf('%s\\textbf{%s}	& %.3f & %d	& %.3f & %.3f	& %.4f & %.3f \\\\', ...
        outString, textlb, tbl{j+1, 2}, tbl{j+1, 3}, tbl{j+1, 5}, ...
        tbl{j+1, 6}, tbl{j+1, 7}, soa.omega2p.(lb));

        %fprintf(fid, '\\hline \n');
        outString = sprintf('%s\n', outString);
    end

    outString = sprintf('%s\\textbf{Error} & %.3f & %d & %.3f & & &  \\\\', ...
    outString, tbl{end-1, 2}, tbl{end-1, 3}, tbl{end-1, 5});

    outString = sprintf('%s\n\\hline\n', outString);

    outString = sprintf('%s\\textbf{Total} & %.3f & %d & & & &  \\\\', ...
    outString, tbl{end, 2}, tbl{end, 3});

    outString = sprintf('%s\n\\hline\n', outString);

    outString = sprintf('%s\\end{tabular} \n', outString);
    outString = sprintf('%s}', outString);
    outString = sprintf('%s\\end{table} \n\n', outString);

end

