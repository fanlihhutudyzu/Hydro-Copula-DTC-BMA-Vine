%% scenario_loop_vine_bma_4d.m
% Run each scenario once (synthetic sample 1000x4), BMA simulation 1000x4, and output exhaustive results and plots
% Oct 18 2025
clear; clc; close all;
rng(1); % reproducibility
%% ========== Settings ==========
scenarios = {'A','B','C'};    % A: t-copula tail, B: mixed arch+gauss blocks, C: mixture-per-observation
n = 1000;                     % Fixed synthetic sample size 1000 x 4
d = 4;                        % Dimensions
Nsim = 1000;                  % Simulation sample size (1000 x 4)
%% families - Modified to 5 distributions commonly used in hydrology
marg_families = {'Gamma','Lognormal','Pearson3','GEV','Weibull'};  
copula_families = {'Gaussian','t','Clayton','Gumbel','Frank'};
%% Graphic style settings
set(groot, 'DefaultLineLineWidth', 1.2);
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Times New Roman');
set(groot, 'DefaultTextFontSize', 10);
set(groot, 'DefaultLegendFontSize', 9);
%% Scatter plot style parameters (transparency settings)
original_color = [0 0.447 0.741];    % Blue (RGB triplet)
simulated_color = [0.85 0.333 0.1];  % Orange (RGB triplet)
alpha_value = 0.3;                   % Transparency (0-1)
original_marker_size = 30;
simulated_marker_size = 30;          % Uniform size for comparison
%% ========== Process three scenarios one by one ==========
for sIdx = 1:length(scenarios)
    scen = scenarios{sIdx};
    fprintf('\n====== Scenario %s ======\n', scen);
    try
        %% 1) Generate 1000x4 synthetic samples
        data = generate_scenario_data(scen, n); % Returns n x 4
        
        %% 2) Marginal fitting (BIC selection) & K-S test & Pseudo-observations Uhat
        marg_fits = cell(1,d);
        KS_pvals = nan(1,d);
        KS_results = cell(1,d);  % Store test results
        Uhat = zeros(n,d);
        for i = 1:d
            x = data(:,i);
            bestBIC = Inf;
            bestFit = [];
            bestFam = '';
            
            % Fit all candidate distributions and select the optimal BIC model
            for f = marg_families
                fam = f{1};
                try
                    switch fam
                        case {'Gamma', 'Lognormal', 'Weibull'}
                            pd = fitdist(x, fam);  % Built-in distributions
                            p = 2;
                        case 'Pearson3'
                            pd = pearson3fit(x);  % Custom distribution
                            p = 3;
                        case 'GEV'  % GEV distribution processing
                            [shape, loc, scale] = gevfit(x);
                            pd = struct('shape', shape, 'loc', loc, 'scale', scale);
                            p = 3;
                    end
                    
                    % Calculate BIC
                    if strcmp(fam, 'GEV')
                        pdfv = gevpdf(x, pd.shape, pd.loc, pd.scale);
                    elseif isstruct(pd) && isfield(pd, 'pdf')
                        pdfv = pd.pdf(x);
                    else
                        pdfv = pdf(pd, x);
                    end
                    pdfv(pdfv <= 0) = realmin;
                    logL = sum(log(pdfv));
                    BIC = -2 * logL + p * log(n);
                    
                    % Update optimal model
                    if BIC < bestBIC
                        bestBIC = BIC;
                        bestFit = pd;
                        bestFam = fam;
                    end
                catch
                    continue;
                end
            end
            
            marg_fits{i}.pd = bestFit;
            marg_fits{i}.family = bestFam;
            
            % Calculate pseudo-observations Uhat and K-S test
            if ~isempty(bestFit)
                try
                    if strcmp(bestFam, 'GEV')
                        cdf_func = @(t) gevcdf(t, bestFit.shape, bestFit.loc, bestFit.scale);
                    elseif isstruct(bestFit) && isfield(bestFit, 'cdf')
                        cdf_func = @(t) bestFit.cdf(bestFit, t);
                    else
                        cdf_func = @(t) cdf(bestFit, t);
                    end
                    
                    x_min = min(x);
                    x_max = max(x);
                    t = linspace(x_min - 0.1*(x_max-x_min), x_max + 0.1*(x_max-x_min), 1000);
                    cdf_vals = cdf_func(t);
                    cdf_matrix = [t', cdf_vals'];
                    
                    if size(cdf_matrix, 2) ~= 2 || any(isnan(cdf_matrix(:)))
                        error('CDF matrix construction failed');
                    end
                    
                    [h, pval] = kstest(x, 'CDF', cdf_matrix, 'Alpha', 0.05);
                    Uhat(:,i) = cdf_func(x);
                    KS_pvals(i) = pval;
                    KS_results{i} = sprintf('%s (p=%.4f)', (h==0)*'Passed' + (h==1)*'Failed', pval);
                catch ME
                    warning('K-S test failed for variable %d: %s', i, ME.message);
                    Uhat(:,i) = tiedrank(x)/(n+1);
                    KS_pvals(i) = NaN;
                    KS_results{i} = 'Error';
                end
            else
                Uhat(:,i) = tiedrank(x)/(n+1);
                KS_pvals(i) = NaN;
                KS_results{i} = 'No fit';
            end
        end
        Uhat = min(max(Uhat, 1e-6), 1-1e-6);
        Marginal_KS_Pvalues = KS_pvals;
        
        %% 3) Generate unique C-vine and D-vine orders
        orders_all = perms(1:d);
        unique_C_orders = {};
        unique_D_orders = {};
        % C-vine uniqueness (swap tail)
        for ii=1:size(orders_all,1)
            ord = orders_all(ii,:);
            ord_swap_tail = ord;
            if d>2
                ord_swap_tail([d-1,d]) = ord_swap_tail([d,d-1]);
            end
            is_C_unique = true;
            for k = 1:length(unique_C_orders)
                if isequal(ord, unique_C_orders{k}) || isequal(ord_swap_tail, unique_C_orders{k})
                    is_C_unique = false; break;
                end
            end
            if is_C_unique, unique_C_orders{end+1} = ord; end
        end
        % D-vine uniqueness (reverse)
        for ii=1:size(orders_all,1)
            ord = orders_all(ii,:);
            ord_rev = flip(ord);
            is_D_unique = true;
            for k = 1:length(unique_D_orders)
                if isequal(ord, unique_D_orders{k}) || isequal(ord_rev, unique_D_orders{k})
                    is_D_unique = false; break;
                end
            end
            if is_D_unique, unique_D_orders{end+1} = ord; end
        end
        
        %% 4) Fit all unique C-vines and D-vines
        unique_C_models = cell(1, length(unique_C_orders));
        unique_D_models = cell(1, length(unique_D_orders));
        bestC.BIC = Inf; bestD.BIC = Inf;
        % Fit C-vines
        for ii = 1:length(unique_C_orders)
            ord = unique_C_orders{ii};
            try
                modelC = fit_four_dim_cvine(Uhat, ord, copula_families);
                modelC.type = 'C';
                unique_C_models{ii} = modelC;
                if isfield(modelC,'BIC') && modelC.BIC < bestC.BIC
                    bestC = modelC;
                end
            catch
                unique_C_models{ii} = [];
            end
        end
        % Fit D-vines
        for ii = 1:length(unique_D_orders)
            ord = unique_D_orders{ii};
            try
                modelD = fit_four_dim_dvine(Uhat, ord, copula_families);
                modelD.type = 'D';
                unique_D_models{ii} = modelD;
                if isfield(modelD,'BIC') && modelD.BIC < bestD.BIC
                    bestD = modelD;
                end
            catch
                unique_D_models{ii} = [];
            end
        end
        
        % Merge models and collect available model information
        all_models = [unique_C_models, unique_D_models];
        model_info = struct('type', {}, 'order', {}, 'BIC', {}, 'model', {});
        for k = 1:length(all_models)
            mm = all_models{k};
            if ~isempty(mm) && isfield(mm,'BIC')
                model_info(end+1).type = mm.type;             
                model_info(end).order = mm.order;
                model_info(end).BIC = mm.BIC;
                model_info(end).model = mm;
            end
        end
        Num_Valid_Models = length(model_info);
        if Num_Valid_Models == 0
            error('No Vine models were successfully fitted');
        end
        
        % Extract BIC list and orders/types
        allBICs = [model_info.BIC];
        allOrders = cellfun(@(o) sprintf('%d-', o{:}), arrayfun(@(m) {m.order}, model_info, 'UniformOutput', false), 'UniformOutput', false);
        % Construct strings
        AllOrders_strs = cell(1,Num_Valid_Models);
        AllTypes = cell(1,Num_Valid_Models);
        for ii=1:Num_Valid_Models
            ordv = model_info(ii).order;
            AllOrders_strs{ii} = sprintf('%d-', ordv);
            AllOrders_strs{ii}(end) = []; % remove trailing '-'
            AllTypes{ii} = model_info(ii).type;
        end
        
        % Calculate BMA weights (based on BIC) - Complete the definition of allW
        logL_rel = -0.5 * allBICs;
        logL_rel = logL_rel - max(logL_rel); % Stabilization
        w_num = exp(logL_rel);
        allW = w_num / sum(w_num);  % Define allW here
        
        % Find the best C and D (if they exist)
        bestC_order_str = 'N/A'; bestD_order_str = 'N/A';
        BIC_bestC = NaN; BIC_bestD = NaN;
        C_idxs = find(strcmp({model_info.type}, 'C'));
        D_idxs = find(strcmp({model_info.type}, 'D'));
        if ~isempty(C_idxs)
            [BIC_bestC, posC] = min(allBICs(C_idxs));
            bestC_info = model_info(C_idxs(posC));
            bestC_order_str = sprintf('%d-%d-%d-%d', bestC_info.order);
        end
        if ~isempty(D_idxs)
            [BIC_bestD, posD] = min(allBICs(D_idxs));
            bestD_info = model_info(D_idxs(posD));
            bestD_order_str = sprintf('%d-%d-%d-%d', bestD_info.order);
        end
        
        % Overall best model (by BIC)
        [minBIC, minPos] = min(allBICs);
        BestModelType = model_info(minPos).type;
        BestOrder_str = sprintf('%d-%d-%d-%d', model_info(minPos).order);
        
        % Result string concatenation
        AllBICs_str = strjoin(arrayfun(@(v) sprintf('%.2f',v), allBICs, 'UniformOutput', false), ';');
        AllW_str   = strjoin(arrayfun(@(v) sprintf('%.4f',v), allW, 'UniformOutput', false), ';');
        AllOrders_str = strjoin(AllOrders_strs, ';');
        AllTypes_str  = strjoin(AllTypes, ';');
        KS_pvals_str  = strjoin(arrayfun(@(v) sprintf('%.4f',v), Marginal_KS_Pvalues, 'UniformOutput', false), ';');
        KS_results_str = strjoin(KS_results, ';');  % K-S test results string
        
        %% 5) Simulation based on BMA weights
        simU = zeros(Nsim, d);
        Nk_raw = round(Nsim .* allW);
        diffN = Nsim - sum(Nk_raw);
        if diffN ~= 0
            [~, imax] = max(allW);
            Nk_raw(imax) = Nk_raw(imax) + diffN;
        end
        
        simU_full = [];
        for ii = 1:Num_Valid_Models
            Nk = Nk_raw(ii);
            if Nk <= 0, continue; end
            modelk = model_info(ii).model;
            if strcmp(modelk.type, 'C')
                simk = simulate_four_dim_cvine_fixed(Nk, modelk);
            else
                simk = simulate_four_dim_dvine_fixed(Nk, modelk);
            end
            simU_full = [simU_full; simk]; 
        end
        if size(simU_full,1) >= Nsim
            simU = simU_full(1:Nsim, :);
        else
            extra = Nsim - size(simU_full,1);
            simU = [simU_full; rand(extra,d)];
        end
        
        % Inverse transform to original space
        simBMA_data = zeros(Nsim, d);
        for i=1:d
            pd = marg_fits{i}.pd;
            fam = marg_fits{i}.family;
            uvec = simU(:,i);
            if isempty(pd)
                simBMA_data(:,i) = quantile(data(:,i), uvec);
            else
                if strcmp(fam, 'GEV')
                    simBMA_data(:,i) = gevinv(uvec, pd.shape, pd.loc, pd.scale);
                elseif isstruct(pd) && isfield(pd,'icdf')
                    simBMA_data(:,i) = pd.icdf(uvec);
                else
                    simBMA_data(:,i) = icdf(pd, uvec);
                end
            end
        end
        
        %% 6) Display results
        fprintf('Scenario %s results:\n', scen);
        fprintf('  Best overall model: %s (order %s) with BIC=%.3f\n', BestModelType, BestOrder_str, minBIC);
        fprintf('  Best C-vine: %s  Best D-vine: %s\n', bestC_order_str, bestD_order_str);
        fprintf('  Num valid vine models: %d\n', Num_Valid_Models);
        fprintf('  All orders: %s\n', AllOrders_str);
        fprintf('  All types : %s\n', AllTypes_str);
        fprintf('  All BICs  : %s\n', AllBICs_str);
        fprintf('  All weights: %s\n', AllW_str);
        fprintf('  Marginal KS test results: %s\n', KS_results_str);
        fprintf('  Marginal KS p-values: %s\n', KS_pvals_str);
        
        % Correlation matrices
        R_orig = corr(data, 'Type', 'Pearson');
        R_sim  = corr(simBMA_data, 'Type', 'Pearson');
        disp('  Pearson correlation matrix (original):'); disp(R_orig);
        disp('  Pearson correlation matrix (simulated):'); disp(R_sim);
        
        Tau_orig = corr(Uhat, 'Type', 'Kendall');
        Tau_sim  = corr(simU, 'Type', 'Kendall');
        disp('  Kendall correlation matrix (original):'); disp(Tau_orig);
        disp('  Kendall correlation matrix (simulated):'); disp(Tau_sim);
        
        %% 8) Plotting
        fig = figure('Name', ['Scenario ', scen], 'Units','normalized','Position',[0.1 0.1 0.9 0.7]);
        fig.Color = [1 1 1]; % White background
        pairs = nchoosek(1:d,2);
        
        % QQ Plots (First 4)
        for i=1:4
            subplot(2,5,i);
            x = data(:,i);
            pd = marg_fits{i}.pd;
            fam = marg_fits{i}.family;
            nobs = length(x);
            p = (1:nobs)'/(nobs+1);
            
            % Calculate theoretical quantiles
            if ~isempty(pd)
                if strcmp(fam, 'GEV')
                    q_theo = gevinv(p, pd.shape, pd.loc, pd.scale);
                elseif isstruct(pd) && isfield(pd,'icdf')
                    q_theo = pd.icdf(p);
                else
                    q_theo = icdf(pd, p);
                end
            else
                q_theo = quantile(x, p);
            end
            q_emp = sort(x);
            
            % Draw QQ Plot
            plot(q_theo, q_emp, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', [0.2 0.2 0.8], 'MarkerFaceColor', [0.7 0.7 1]);
            hold on;
            mn = min(min(q_theo), min(q_emp)); 
            mx = max(max(q_theo), max(q_emp));
            plot([mn mx], [mn mx], 'r--', 'LineWidth', 1.2);
            xlabel(sprintf('Theoretical Q (%s)', fam), 'FontWeight', 'bold');
            ylabel('Sample Q', 'FontWeight', 'bold');
            title(sprintf('Variable %d', i), 'FontWeight', 'bold');
            grid on;
            box on;
            axis equal;
        end
        
        % Scatter Comparisons (Remaining 6)
        for k=1:6
            subplot(2,5,4+k);
            i = pairs(k,1); j = pairs(k,2);
            
            % Original data scatter (Semi-transparent blue)
            scatter(data(:,i), data(:,j), original_marker_size, ...
                original_color, 'filled', ...
                'MarkerEdgeColor', 'none', ...
                'MarkerFaceAlpha', alpha_value);
            
            hold on;
            
            % Simulated data scatter (Semi-transparent orange)
            scatter(simBMA_data(:,i), simBMA_data(:,j), simulated_marker_size, ...
                simulated_color, 'filled', ...
                'MarkerEdgeColor', 'none', ...
                'MarkerFaceAlpha', alpha_value);
            
            xlabel(sprintf('X%d', i), 'FontWeight', 'bold');
            ylabel(sprintf('X%d', j), 'FontWeight', 'bold');
            title(sprintf('X%d vs X%d', i, j), 'FontWeight', 'bold');
            legend({'Original', 'Simulated'}, 'Location', 'best', 'Box', 'off'); 
            grid on;
            box on;
        end
        
        % Add overall title
        sgtitle(sprintf('Scenario %s: Marginal QQ Plots and Scatter Comparisons', scen), ...
            'FontSize', 14, 'FontWeight', 'bold');
        
                
    catch ME
        warning('Scenario %s failed: %s', scen, ME.message);
    end
end
% Reset graphic default settings
set(groot, 'DefaultLineLineWidth', 0.5);
set(groot, 'DefaultAxesFontName', 'Helvetica');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Helvetica');
set(groot, 'DefaultTextFontSize', 10);
set(groot, 'DefaultLegendFontSize', 9);