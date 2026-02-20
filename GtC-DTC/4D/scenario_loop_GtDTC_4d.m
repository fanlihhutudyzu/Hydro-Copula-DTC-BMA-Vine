%% scenario_loop_GtDTC_4d.m

clear; clc; close all;
rng(1);

%% ========== Settings ==========
scenarios = {'A','B','C'};
n = 1000;                     % Synthetic sample size
d = 4;                        % Dimension
Nsim = 1000;                  % Number of simulated samples

% Manual method selection
fprintf('Please select simulation method:\n');
fprintf('1 - Gaussian/t Copula method\n');
fprintf('2 - DTC method\n');
method_choice = input('Enter method number (1 or 2): ');
while ~ismember(method_choice, [1,2])
    method_choice = input('Invalid input, please re-enter method number (1 or 2): ');
end

%% Marginal distribution families
marg_families = {'Gamma','Lognormal','Pearson3','GEV','Weibull'};  

%% Plot style settings
set(groot, 'DefaultLineLineWidth', 1.2);
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Times New Roman');
set(groot, 'DefaultTextFontSize', 10);
set(groot, 'DefaultLegendFontSize', 9);

%% Scatter plot style parameters
original_color = [0 0.447 0.741];
simulated_color = [0.85 0.333 0.1];
alpha_value = 0.3;
original_marker_size = 30;
simulated_marker_size = 30;

%% ========== Process scenarios ==========
for sIdx = 1:length(scenarios)
    scen = scenarios{sIdx};
    fprintf('\n====== Scenario %s ======\n', scen);
    try
        %% 1) Generate synthetic data
        data = generate_scenario_data(scen, n);
        
        %% 2) Marginal fitting & KS test & pseudo-observations Uhat
        marg_fits = cell(1,d);
        KS_pvals = nan(1,d);
        KS_results = cell(1,d);
        Uhat = zeros(n,d);

        for i = 1:d
            x = data(:,i);
            bestBIC = Inf;
            bestFit = [];
            bestFam = '';
            
            for f = marg_families
                fam = f{1};
                try
                    switch fam
                        case {'Gamma', 'Lognormal', 'Weibull'}
                            pd = fitdist(x, fam);
                            p = 2;
                        case 'Pearson3'
                            pd = pearson3fit(x);
                            p = 3;
                        case 'GEV'
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
            
            % Calculate pseudo-observations and KS test
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
                    warning('KS test failed for variable %d: %s', i, ME.message);
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
        KS_results_str = strjoin(KS_results, ';');
        KS_pvals_str = strjoin(arrayfun(@(v) sprintf('%.4f',v), Marginal_KS_Pvalues, 'UniformOutput', false), ';');
        
        %% 3) Model fitting and simulation
        if method_choice == 1
            [simU, model_info] = gaussian_t_copula_model(Uhat, Nsim);
            method_name = 'Gaussian/t Copula';
        else
            [simU, model_info] = copula_gibbs_model(Uhat, Nsim);
            method_name = 'DTC';
        end
        
        %% 4) Inverse transform to original space
        sim_data = zeros(Nsim, d);
        for i=1:d
            pd = marg_fits{i}.pd;
            fam = marg_fits{i}.family;
            uvec = simU(:,i);
            if isempty(pd)
                sim_data(:,i) = quantile(data(:,i), uvec);
            else
                if strcmp(fam, 'GEV')
                    sim_data(:,i) = gevinv(uvec, pd.shape, pd.loc, pd.scale);
                elseif isstruct(pd) && isfield(pd,'icdf')
                    sim_data(:,i) = pd.icdf(uvec);
                else
                    sim_data(:,i) = icdf(pd, uvec);
                end
            end
        end
        
        %% 5) Display results
        fprintf('Scenario %s results (using method: %s):\n', scen, method_name);
        
        if method_choice == 1
            fprintf('  Selected copula type: %s\n', model_info.chosen_type);
            fprintf('  Gaussian Copula BIC: %.3f\n', model_info.BIC_Gauss);
            fprintf('  t Copula BIC: %.3f\n', model_info.BIC_t);
            if strcmp(model_info.chosen_type, 't')
                fprintf('  t Copula degrees of freedom: %.2f\n', model_info.nu_t);
            end
        else
            fprintf('  Variable sampling order: %s\n', num2str(model_info.var_order));
            fprintf('  Optimal pairwise copula types:\n');
            for k = 1:size(model_info.copula_pairs,1)
                fprintf('    Variable pair (%d,%d): %s\n', ...
                    model_info.copula_pairs(k,1), ...
                    model_info.copula_pairs(k,2), ...
                    model_info.opt_types{k});
            end
        end
        
        fprintf('  Marginal KS test results: %s\n', KS_results_str);
        fprintf('  Marginal KS p-values: %s\n', KS_pvals_str);
        
        % Correlation matrices
        R_orig = corr(data, 'Type', 'Pearson');
        R_sim  = corr(sim_data, 'Type', 'Pearson');
        disp('  Original data Pearson correlation matrix:'); disp(R_orig);
        disp('  Simulated data Pearson correlation matrix:'); disp(R_sim);
        
        Tau_orig = corr(Uhat, 'Type', 'Kendall');
        Tau_sim  = corr(simU, 'Type', 'Kendall');
        disp('  Original data Kendall correlation matrix:'); disp(Tau_orig);
        disp('  Simulated data Kendall correlation matrix:'); disp(Tau_sim);
        
        %% 6) Plotting
        fig = figure('Name', ['Scenario ', scen, ' (', method_name, ')'], ...
            'Units','normalized','Position',[0.1 0.1 0.9 0.7]);
        fig.Color = [1 1 1];
        pairs = nchoosek(1:d,2);
        
        % QQ plots (first 4)
        for i=1:4
            subplot(2,5,i);
            x = data(:,i);
            pd = marg_fits{i}.pd;
            fam = marg_fits{i}.family;
            nobs = length(x);
            p = (1:nobs)'/(nobs+1);
            
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
            
            plot(q_theo, q_emp, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', [0.2 0.2 0.8], 'MarkerFaceColor', [0.7 0.7 1]);
            hold on;
            mn = min(min(q_theo), min(q_emp)); 
            mx = max(max(q_theo), max(q_emp));
            plot([mn mx], [mn mx], 'r--', 'LineWidth', 1.2);
            xlabel(sprintf('Theoretical quantiles (%s)', fam), 'FontWeight', 'bold');
            ylabel('Sample quantiles', 'FontWeight', 'bold');
            title(sprintf('Variable %d', i), 'FontWeight', 'bold');
            grid on;
            box on;
            axis equal;
        end
        
        % Scatter comparisons (last 6)
        for k=1:6
            subplot(2,5,4+k);
            i = pairs(k,1); j = pairs(k,2);
            
            scatter(data(:,i), data(:,j), original_marker_size, ...
                original_color, 'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', alpha_value);
            hold on;
            scatter(sim_data(:,i), sim_data(:,j), simulated_marker_size, ...
                simulated_color, 'filled', 'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', alpha_value);
            
            xlabel(sprintf('X%d', i), 'FontWeight', 'bold');
            ylabel(sprintf('X%d', j), 'FontWeight', 'bold');
            title(sprintf('X%d vs X%d', i, j), 'FontWeight', 'bold');
            legend({'Original data', 'Simulated data'}, 'Location', 'best', 'Box', 'off'); 
            grid on;
            box on;
        end
        
        sgtitle(sprintf('Scenario %s: Marginal QQ Plots and Scatter Comparisons (%s)', scen, method_name), ...
            'FontSize', 14, 'FontWeight', 'bold');
        
        method_suffix = ['_method', num2str(method_choice)];
        print(fig, sprintf('Scenario_%s%s_plot.png', scen, method_suffix), '-dpng', '-r300');
        fprintf('Plot saved as Scenario_%s%s_plot.png\n', scen, method_suffix);
        
    catch ME
        warning('Scenario %s processing failed: %s', scen, ME.message);
    end
end

% Reset figure default settings
set(groot, 'DefaultLineLineWidth', 0.5);
set(groot, 'DefaultAxesFontName', 'Helvetica');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Helvetica');
set(groot, 'DefaultTextFontSize', 10);
set(groot, 'DefaultLegendFontSize', 9);

disp('All scenarios processed. Plots saved as PNG files.');

%% Helper functions
function data = generate_scenario_data(scenario, n)
    d = 4;
    marg = cell(1,d);
    marg{1} = makedist('Gamma','a',2.3,'b',2.1);
    marg{2} = makedist('Gamma','a',2.5,'b',1.9);
    marg{3} = makedist('Gamma','a',2.1,'b',2.2);
    marg{4} = makedist('Gamma','a',2.4,'b',1.8);

    switch scenario
        case 'A'
            R = [1 .6 .35 .2; .6 1 .4 .25; .35 .4 1 .35; .2 .25 .35 1];
            nu = 3;
            U = copularnd('t', R, nu, n);
        case 'B'
            U12 = copularnd('Clayton', 2.2, n);
            U34 = copularnd('Gumbel', 2.5, n);
            U = zeros(n,4);
            U(:,1:2) = U12;
            U(:,3:4) = U34;
            mix_p = 0.25;
            idx_mix = rand(n,1) < mix_p;
            if any(idx_mix)
                Rweak = eye(4); Rweak(1,3)=0.18; Rweak(3,1)=0.18; Rweak(1,4)=0.12; Rweak(4,1)=0.12; Rweak(2,3)=0.1; Rweak(3,2)=0.1;
                Ug = copularnd('Gaussian', Rweak, sum(idx_mix));
                U(idx_mix,:) = Ug;
            end
        case 'C'
            U = zeros(n,4);
            for i=1:n
                r = rand();
                if r <= 0.5
                    Rg = [1 .5 .25 .2; .5 1 .35 .25; .25 .35 1 .3; .2 .25 .3 1];
                    U(i,:) = copularnd('Gaussian', Rg, 1);
                elseif r <= 0.8
                    Rt = [1 .6 .3 .2; .6 1 .4 .25; .3 .4 1 .35; .2 .25 .35 1];
                    U(i,:) = copularnd('t', Rt, 4, 1);
                else
                    c12 = copularnd('Clayton', 2, 1);
                    c34 = copularnd('Gumbel', 2.2, 1);
                    U(i,:) = [c12(1), c12(2), c34(1), c34(2)];
                end
            end
        otherwise
            error('Unknown scenario');
    end

    data = zeros(n,d);
    for j=1:d
        data(:,j) = icdf(marg{j}, U(:,j));
    end
end

%% Method 1: Gaussian/t Copula model
function [sim_data, model_info] = gaussian_t_copula_model(U, m)
[n,d] = size(U);

% Fit Gaussian copula
R_gauss = copulafit('Gaussian', U);
ll_gauss = sum(log(copulapdf('Gaussian', U, R_gauss)));
k_gauss = d*(d-1)/2;
BIC_Gauss = -2*ll_gauss + k_gauss*log(n);

% Fit t copula
try
    [R_t, nu_t] = copulafit('t', U);
catch ME
    warning(ME.identifier,'t copula fitting failed: %s. Using default nu=4 and estimating R via inversion.', ME.message);
    nu_t = 4;
    R_t = copulafit('Gaussian', U);
end

ll_t = sum(log(copulapdf('t', U, R_t, nu_t)));
k_t = d*(d-1)/2 + 1;
BIC_t = -2*ll_t + k_t*log(n);

% Select best model
if BIC_t < BIC_Gauss
    chosen = 't';
    params = struct('R', R_t, 'nu', nu_t);
else
    chosen = 'Gaussian';
    params = struct('R', R_gauss);
end

% Generate simulated samples
if strcmp(chosen,'t')
    Usim = copularnd('t', params.R, params.nu, m);
else
    Usim = copularnd('Gaussian', params.R, m);
end

sim_data = Usim;
model_info = struct('R_gauss',R_gauss,'R_t',R_t,'nu_t',nu_t, ...
    'BIC_Gauss',BIC_Gauss,'BIC_t',BIC_t,'chosen_type',chosen,'chosen_params',params);
end

%% Method 2: DTC model
function [gibbs_sim_data, model_info] = copula_gibbs_model(U, m)
[n,d] = size(U);
if d ~= 4
    error('This method is designed for 4-dimensional data (X1,X2,X3,X4) only');
end

corr_mat = corr(U,'type','Pearson');
[copula_pairs, var_order] = get_copula_pairs_and_order(corr_mat);

candidate_types = {'Gaussian','t','Frank','Gumbel','Clayton'};
num_pairs = size(copula_pairs, 1);
opt_types = cell(1,num_pairs);
opt_params = cell(1,num_pairs);

for k = 1:num_pairs
    pair_idx = copula_pairs(k,:);
    upair = U(:,pair_idx);
    [best_type, best_params] = fit_pairwise_copula(upair, candidate_types);
    opt_types{k} = best_type;
    opt_params{k} = best_params;
end

Usim = zeros(m,d);
sampling_steps = cell(1, d);
sampling_steps{1} = struct('var_sim', var_order(1), 'type', 'marginal');

for j = 2:d
    var_sim = var_order(j);
    simulated_vars = var_order(1:j-1); 
    
    found = false;
    var_cond = -1;
    pair_k = -1;
    
    for cond_cand_idx = 1:length(simulated_vars)
        cond_cand = simulated_vars(cond_cand_idx);
        for k = 1:num_pairs
            pair_current = copula_pairs(k,:);
            if all(ismember([var_sim, cond_cand], pair_current))
                var_cond = cond_cand;
                pair_k = k;
                found = true;
                break;
            end
        end
        if found
            break;
        end
    end

    if ~found
        error('Could not find a copula pair connecting X%d to any previously simulated variables (%s)', ...
              var_sim, mat2str(simulated_vars));
    end
    
    if copula_pairs(pair_k,1) == var_cond
        idx_in_pair_cond = 1;
    elseif copula_pairs(pair_k,2) == var_cond
        idx_in_pair_cond = 2;
    else
        error('Logical error in finding conditioning variable index');
    end
    
    sampling_steps{j} = struct('var_sim', var_sim, 'var_cond', var_cond, ...
                               'pair_idx', pair_k, 'idx_in_pair_cond', idx_in_pair_cond);
end

for i = 1:m
    s = rand(1,d);
    u_current = zeros(1,d);
    u_current(var_order(1)) = s(1);
    
    for j = 2:d
        step = sampling_steps{j};
        pair_k = step.pair_idx;
        var_cond = step.var_cond;
        var_sim = step.var_sim;
        idx_in_pair_cond = step.idx_in_pair_cond;
        
        typ = opt_types{pair_k};
        params = opt_params{pair_k};
        u_prev = u_current(var_cond);
        w = s(j);

        if idx_in_pair_cond == 1
            v = hinv_conditional(typ, params, u_prev, w, 1);
        elseif idx_in_pair_cond == 2
            v = hinv_conditional(typ, params, u_prev, w, 2);
        else
            error('Invalid conditioning index');
        end

        u_j = min(max(v, 1e-12), 1-1e-12);
        u_current(var_sim) = u_j;
    end
    Usim(i,:) = u_current;
end

gibbs_sim_data = Usim; 
model_info = struct('corr_mat',corr_mat,'var_order',var_order, ...
    'copula_pairs',copula_pairs, 'opt_types',{opt_types},'opt_params',{opt_params});
end

%% Other helper functions
function [copula_pairs, var_order] = get_copula_pairs_and_order(corr_mat)
d = size(corr_mat,1);
abs_corr = abs(corr_mat);
abs_corr(1:d+1:end) = 0;

pairs_list = [];
for i = 1:d
    for j = i+1:d
        pairs_list = [pairs_list; abs_corr(i,j), i, j];
    end
end
pairs_list = sortrows(pairs_list, 1, 'descend');

r_max_pair = pairs_list(1, 2:3);
r_sec_pair = pairs_list(2, 2:3);
overlap = intersect(r_max_pair, r_sec_pair);

if ~isempty(overlap)
    mid_node = overlap(1); 
    other_in_max = setdiff(r_max_pair, mid_node);
    pair1 = sort([mid_node, other_in_max]);
    other_in_sec = setdiff(r_sec_pair, mid_node);
    pair2 = sort([mid_node, other_in_sec]);
    remaining_var = setdiff(1:d, [pair1, pair2]);
    if isempty(remaining_var)
        error('Case 1 error: All variables covered by r_max and r_sec');
    end
    remaining_var = remaining_var(1); 
    all_other_vars = [mid_node, other_in_max, other_in_sec];
    corr_to_others = abs_corr(remaining_var, all_other_vars);
    [~, max_corr_idx] = max(corr_to_others);
    third_node = all_other_vars(max_corr_idx);
    pair3 = sort([remaining_var, third_node]);
    copula_pairs = unique([pair1; pair2; pair3], 'rows'); 
    X_start = mid_node;
    X_end = remaining_var;
    X_connect_to_end = third_node;
    X_other_mid_connect = setdiff([other_in_max, other_in_sec], X_connect_to_end);
    if isempty(X_other_mid_connect) && X_connect_to_end == X_start
        X_other_mid_connect = setdiff(1:d, [X_start, X_end]); 
        var_order = [X_start, X_other_mid_connect, X_end]; 
    elseif isempty(X_other_mid_connect)
        error('Error determining variable order for Case 1');
    else
        var_order = [X_start, X_connect_to_end, X_other_mid_connect, X_end];
    end
else
    pair1 = sort(r_max_pair);
    pair2 = sort(r_sec_pair);
    r_third_pair = pairs_list(3, 2:3);
    pair3 = sort(r_third_pair);
    copula_pairs = unique([pair1; pair2; pair3], 'rows');
    all_vars = [pair1, pair2, pair3];
    [counts, values] = hist(all_vars, unique(all_vars));
    endpoints = values(counts == 1);
    middle_nodes = values(counts == 2);
    var_order = zeros(1, d);
    if length(endpoints) == 2 
        var_order(1) = endpoints(1);
        used = [endpoints(1)];
        current_var = var_order(1);
        for i = 2:d
            next_var = -1;
            for k = 1:size(copula_pairs, 1)
                p = copula_pairs(k,:);
                if ismember(current_var, p)
                    next_cand = setdiff(p, current_var);
                    if ~ismember(next_cand, used) && next_cand ~= -1
                        next_var = next_cand;
                        break;
                    end
                end
            end
            if next_var == -1
                remaining = setdiff(1:d, used);
                if ~isempty(remaining)
                    next_var = remaining(1);
                else
                    break;
                end
            end
            var_order(i) = next_var;
            used = [used, next_var];
            current_var = next_var;
        end
    else
        var_order = 1:d; 
    end
    if any(var_order == 0)
        var_order = [var_order(var_order ~= 0), setdiff(1:d, var_order(var_order ~= 0))];
    end
end
end

function v = hinv_conditional(type, params, u_cond, w, idx_in_pair_cond)
if idx_in_pair_cond == 1
    if strcmpi(type,'Gaussian')
        rho = params.R(1,2);
        v = hinv_gaussian(w, u_cond, rho); 
    elseif strcmpi(type,'t')
        rho = params.R(1,2);
        nu = params.nu;
        v = hinv_t(w, u_cond, rho, nu); 
    else
        f = @(v) numeric_partial_h(type, params, u_cond, v, 1) - w;
        v = hinv_numeric(f);
    end
elseif idx_in_pair_cond == 2
    if strcmpi(type,'Gaussian')
        rho = params.R(1,2);
        v = hinv_gaussian(w, u_cond, rho); 
    elseif strcmpi(type,'t')
        rho = params.R(1,2);
        nu = params.nu;
        v = hinv_t(w, u_cond, rho, nu);
    else
        f = @(v) numeric_partial_h(type, params, v, u_cond, 2) - w;
        v = hinv_numeric(f);
    end
else
    error('Invalid conditioning index');
end
end

function u_j = hinv_numeric(f)
lb = 1e-8; ub = 1-1e-8;
x0 = 0.5; 
try
    if f(lb)*f(ub) > 0
        grid = linspace(lb,ub,9);
        vals = arrayfun(f, grid);
        idx = find(vals(1:end-1).*vals(2:end) < 0, 1);
        if isempty(idx)
            [~, closest_idx] = min(abs(vals));
            x0 = grid(closest_idx);
        else
            lb = grid(idx);
            ub = grid(idx+1);
            x0 = (lb+ub)/2;
        end
    end
    u_j = fzero(f, x0);
catch
    u_j = bisection(f, lb, ub, 1e-8, 100);
end
u_j = min(max(u_j, 1e-12), 1-1e-12);
end

function h = numeric_partial_h(type, params, u, v, w_r_t)
epsu = 1e-6;
if w_r_t == 1
    u1 = min(max(u+epsu, 0), 1);
    u0 = min(max(u-epsu, 0), 1);
    uu1 = [u1 v]; uu0 = [u0 v];
    denom = 2*epsu;
elseif w_r_t == 2
    v1 = min(max(v+epsu, 0), 1);
    v0 = min(max(v-epsu, 0), 1);
    uu1 = [u v1]; uu0 = [u v0];
    denom = 2*epsu;
else
    error('w_r_t must be 1 or 2');
end
try
    if strcmpi(type,'Gaussian')
        C1 = copulacdf('Gaussian', uu1, params.R);
        C0 = copulacdf('Gaussian', uu0, params.R);
    elseif strcmpi(type,'t')
        C1 = copulacdf('t', uu1, params.R, params.nu);
        C0 = copulacdf('t', uu0, params.R, params.nu);
    else
        C1 = copulacdf(type, uu1, params.theta);
        C0 = copulacdf(type, uu0, params.theta);
    end
    h = (C1 - C0) / denom;
catch ME
    warning(ME.identifier,'numeric_partial_h failed: %s', ME.message);
    epsu = 1e-4;
    if w_r_t == 1
        u1 = min(max(u+epsu, 0), 1);
        u0 = min(max(u-epsu, 0), 1);
        uu1 = [u1 v]; uu0 = [u0 v];
    else
        v1 = min(max(v+epsu, 0), 1);
        v0 = min(max(v-epsu, 0), 1);
        uu1 = [u v1]; uu0 = [u v0];
    end
    denom = 2*epsu;
    if strcmpi(type,'Gaussian')
        C1 = copulacdf('Gaussian', uu1, params.R);
        C0 = copulacdf('Gaussian', uu0, params.R);
    elseif strcmpi(type,'t')
        C1 = copulacdf('t', uu1, params.R, params.nu);
        C0 = copulacdf('t', uu0, params.R, params.nu);
    else
        C1 = copulacdf(type, uu1, params.theta);
        C0 = copulacdf(type, uu0, params.theta);
    end
    h = (C1 - C0) / denom;
end
end

function [best_type, best_params] = fit_pairwise_copula(u_pair, candidate_types)
n = size(u_pair,1);
best_BIC = Inf; best_type = ''; best_params = [];
for t = 1:length(candidate_types)
    type = candidate_types{t};
    try
        if strcmpi(type,'t')
            [Rhat, nuhat] = copulafit('t', u_pair);
            ll = sum(log(copulapdf('t', u_pair, Rhat, nuhat)));
            k = 2;
            params = struct('R', Rhat, 'nu', nuhat);
        elseif strcmpi(type,'Gaussian')
            Rhat = copulafit('Gaussian', u_pair);
            ll = sum(log(copulapdf('Gaussian', u_pair, Rhat)));
            k = 1;
            params = struct('R', Rhat);
        else
            theta = copulafit(type, u_pair);
            ll = sum(log(copulapdf(type, u_pair, theta)));
            k = 1;
            params = struct('theta', theta);
        end
        BIC = -2*ll + k*log(n);
        if BIC < best_BIC
            best_BIC = BIC;
            best_type = type;
            best_params = params;
        end
    catch ME
        warning('Failed to fit %s pairwise copula: %s', type, ME.message);
    end
end
if isempty(best_type)
    Rhat = copulafit('Gaussian', u_pair);
    best_type = 'Gaussian';
    best_params = struct('R', Rhat);
end
end

function v = hinv_gaussian(w, u, rho)
z_u = norminv(u);
z_w = norminv(w);
z_v = rho * z_u + sqrt(max(0,1-rho^2)) * z_w;
v = normcdf(z_v);
end

function v = hinv_t(w, u, rho, nu)
z_u = tinv(u, nu);
mean_c = rho * z_u;
scale_c = sqrt( (nu + z_u.^2) / (nu + 1) * (1 - rho^2) );
t_w = tinv(w, nu+1);
t_v = mean_c + scale_c * t_w;
v = tcdf(t_v, nu);
end

function root = bisection(f, a, b, tol, maxit)
fa = f(a); fb = f(b);
if fa*fb > 0
    root = (a+b)/2; return;
end
for k=1:maxit
    c = (a+b)/2;
    fc = f(c);
    if abs(fc) < tol || (b-a)/2 < tol
        root = c; return;
    end
    if fa*fc <= 0
        b = c; fb = fc;
    else
        a = c; fa = fc;
    end
end
root = (a+b)/2;
end