% 3D Multi-scenario Copula Analysis: Synthetic Data from Three Scenarios + Marginal Fitting + Multiple Simulation Methods + Statistical Comparison
clear; clc; close all; 
rng(1);
% Allow user to select simulation method
fprintf('Please select a simulation method:\n');
fprintf('1. Gaussian/t Copula 3D Simulation\n');
fprintf('2. DTC 3D Simulation\n');
method_choice = input('Enter your choice (1 or 2): ');
% Validate user input
while method_choice ~= 1 && method_choice ~= 2
    fprintf('Invalid selection, please try again!\n');
    method_choice = input('Enter your choice (1 or 2): ');
end
% Define scenarios to analyze
scenarios = {'A', 'B', 'C'};
n = 1000;  % Sample size
% Loop through each scenario
for s = 1:length(scenarios)
    scenario = scenarios{s};
    fprintf('\n=============================================\n');
    fprintf('           Analyzing Scenario %s           \n', scenario);
    fprintf('=============================================\n');
    
    % Generate 3D synthetic data for current scenario
    data = generate_scenario_3Ddata(scenario, n);
    d = size(data, 2);  % Dimension fixed at 3
    
    %% 2. Marginal Fitting and Selection (AICc instead of BIC)
    % 5 common distribution functions in hydrology
    families = {'Gamma','Lognormal','Pearson3','GEV','Weibull'};
    marg_fits = cell(1,d); % Pre-allocate cell array
    for i = 1:d
        x = data(:, i);
        bestAICc = Inf;
        for f = families
            fam = f{1};
            try
                switch fam
                    case 'Gamma'
                        pd = fitdist(x,'Gamma'); p = 2;
                    case 'Lognormal'
                        pd = fitdist(x,'Lognormal'); p = 2;
                    case 'Weibull'
                        pd = fitdist(x,'Weibull'); p = 2;
                    case 'Pearson3'
                        pd = pearson3fit(x); p = 3;
                    case 'GEV'
                        pd = gevfit(x); p = 3;  % GEV parameter estimation
                end
                % Calculate log-likelihood and AICc
                if strcmp(fam, 'GEV')
                    pdfv = gevpdf(x, pd(1), pd(2), pd(3));
                elseif isstruct(pd)
                    pdfv = pd.pdf(x);
                else
                    pdfv = pdf(pd, x);
                end
                pdfv(pdfv <= 0) = realmin;
                logL = sum(log(pdfv));
                AIC = -2 * logL + 2*p;
                AICc = AIC + 2*p*(p+1)/(n-p-1);
                
                % Update best model
                if AICc < bestAICc
                    bestAICc = AICc; 
                    bestFit = pd; 
                    bestFam = fam;
                end
            catch
                continue;
            end
        end
        marg_fits{i}.pd = bestFit;
        marg_fits{i}.family = bestFam;
        fprintf('Var%d: best marginal = %s\n', i, bestFam);
    end
    
    %% 3. Get Pseudo-observations 
    Uhat = zeros(n, d);
    for i = 1:d
        x = data(:, i);
        pd = marg_fits{i}.pd;
        fam = marg_fits{i}.family;
        % Calculate CDF based on chosen distribution
        if strcmp(fam, 'GEV')
            Uhat(:, i) = gevcdf(x, pd(1), pd(2), pd(3));
        elseif isstruct(pd)
            Uhat(:, i) = pd.cdf(x);
        else
            Uhat(:, i) = cdf(pd, x);
        end
    end
    % Handle potential extreme values
    Uhat = max(realmin, min(1 - realmin, Uhat));
    
    %% 4. Marginal Diagnostics: QQ Plots and Goodness of Fit
    % Set figure style according to SCI publication requirements
    set(groot, 'DefaultLineLineWidth', 1.2);
    set(groot, 'DefaultAxesFontName', 'Times New Roman');
    set(groot, 'DefaultAxesFontSize', 10);
    set(groot, 'DefaultTextFontName', 'Times New Roman');
    
    figure('Name', sprintf('Scenario %s: Marginal QQ Plots', scenario), 'Position', [100 100 1200 400]);
    for i = 1:d
        subplot(1, 3, i);
        x = data(:, i);
        pd = marg_fits{i}.pd;
        fam = marg_fits{i}.family;
        
        % Calculate theoretical quantiles
        nobs = length(x);
        p = (1:nobs)' / (nobs + 1); % Avoid endpoint issues
        if strcmp(fam, 'GEV')
            q_theo = gevinv(p, pd(1), pd(2), pd(3));
        elseif isstruct(pd)
            q_theo = pd.icdf(p);
        else
            q_theo = icdf(pd, p);
        end
        
        % Sample quantiles
        q_emp = sort(x);
        
        % Plot QQ plot
        plot(q_theo, q_emp, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', [0.2 0.2 0.8], 'MarkerFaceColor', [0.7 0.7 1]);
        hold on;
        % Add reference line y=x
        min_val = min([q_theo; q_emp]);
        max_val = max([q_theo; q_emp]);
        line([min_val, max_val], [min_val, max_val], 'Color', [0.8 0 0], 'LineStyle', '--');
        xlabel(sprintf('Theoretical Quantiles (%s)', fam), 'FontWeight', 'bold');
        ylabel('Sample Quantiles', 'FontWeight', 'bold');
        title(sprintf('Variable %d', i), 'FontWeight', 'bold');
        grid on;
        box on; 
    end
    sgtitle(sprintf('Scenario %s: Marginal Distribution QQ Plots', scenario), 'FontSize', 14, 'FontWeight', 'bold');
    
    %% 5. Simulate based on selected method
    Nsim = 1000;
    sim_data = [];
    model_info = [];
    
    fprintf('\n=== Starting Simulation ===\n');
    if method_choice == 1
        fprintf('Using Gaussian/t Copula 3D simulation method...\n');
        [sim_U, model_info] = gaussian_t_copula_3d_model(Uhat, Nsim);
    else
        fprintf('Using DTC 3D simulation method...\n');
        [sim_U, model_info] = DTC_3d_model(Uhat, Nsim);
    end
    
    % Display model information
    fprintf('Simulation Model Info:\n');
    if method_choice == 1
        fprintf('Chosen Copula Type: %s\n', model_info.chosen_type);
        fprintf('Gaussian AICc: %.3f, t Copula AICc: %.3f\n', model_info.AICc_Gauss, model_info.AICc_t);
    else
        fprintf('Central Node: %d\n', model_info.mid_node);
        fprintf('Optimum Copula Pair 1: %s\n', model_info.opt_types{1});
        fprintf('Optimum Copula Pair 2: %s\n', model_info.opt_types{2});
    end
    
    % Transform back to original data space
    sim_data = zeros(size(sim_U));
    for i = 1:d
        pd = marg_fits{i}.pd;
        fam = marg_fits{i}.family;
        if strcmp(fam, 'GEV')
            sim_data(:, i) = gevinv(sim_U(:, i), pd(1), pd(2), pd(3));
        elseif isstruct(pd)
            sim_data(:, i) = pd.icdf(sim_U(:, i));
        else
            sim_data(:, i) = icdf(pd, sim_U(:, i));
        end
    end
    
    Nsim_actual = size(sim_data, 1);
    fprintf('Simulated sample size: %d\n', Nsim_actual);
    
    %% 6. Compare Statistics 
    fprintf('\n--- Comparing Means and Standard Deviations ---\n');
    for j = 1:d
        fprintf('Var%d mean: orig=%.3f sim=%.3f | std: orig=%.3f sim=%.3f\n',...
            j, mean(data(:, j)), mean(sim_data(:, j)), std(data(:, j)), std(sim_data(:, j)));
    end
    
    %% 7. Compare Correlation Structure 
    R_orig = corrcoef(data);
    R_sim = corrcoef(sim_data);
    fprintf('\n--- Comparing Correlation Structures ---\n');
    fprintf('Original data sample correlation matrix:\n');
    disp(R_orig);
    fprintf('Simulated data sample correlation matrix:\n');
    disp(R_sim);
    
    %% 8. Scatter Plot Comparison 
    pairs = nchoosek(1:d, 2);  % 3D data has 3 variable combinations
    figure('Name', sprintf('Scenario %s: Scatter Plots', scenario), 'Position', [200 200 1200 400]);
    % Define transparency parameters and colors
    alpha_val = 0.3;                  % Transparency (0-1)
    original_color = [0 0.447 0.741]; % Blue (Original)
    simulated_color = [0.85 0.333 0.1];% Orange (Simulated)
    
    for k = 1:size(pairs, 1)
        subplot(1, 3, k);
        i = pairs(k, 1); j = pairs(k, 2);
        % Original scatter (translucent blue)
        scatter(data(:, i), data(:, j), 30, original_color, 'filled', ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', alpha_val);
        
        hold on;
        % Simulated scatter (translucent orange)
        scatter(sim_data(:, i), data(:, j), 30, simulated_color, 'filled', ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', alpha_val);
        
        xlabel(sprintf('X%d', i), 'FontWeight', 'bold'); 
        ylabel(sprintf('X%d', j), 'FontWeight', 'bold');
        title(sprintf('X%d vs X%d', i, j), 'FontWeight', 'bold');
        grid on;
        box on;
        % Add legend to distinguish data types
        if k == 1  % Add legend only in the first subplot
            legend({'Original', 'Simulated'}, 'Location', 'best', 'Box', 'off');
        end
    end
    sgtitle(sprintf('Scenario %s: Original vs Simulated', scenario), 'FontSize', 14, 'FontWeight', 'bold');
end
% Reset graphics default settings
set(groot, 'DefaultLineLineWidth', 0.5);
set(groot, 'DefaultAxesFontName', 'Helvetica');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Helvetica');

%% ===================== Core Method 1: Gaussian/t Copula 3D Simulation (AICc) =====================
function [sim_U, model_info] = gaussian_t_copula_3d_model(Uhat, Nsim)
    [n, d] = size(Uhat);
    if d ~= 3
        error('Input Uhat must be an n x 3 matrix (3D variables), current dimension: n x %d', d);
    end
    if ~(isnumeric(Nsim) && mod(Nsim,1)==0 && Nsim>0)
        error('Nsim must be a positive integer, current value: %s', num2str(Nsim));
    end
    
    R_gauss = copulafit('Gaussian', Uhat);
    ll_gauss = sum(log(copulapdf('Gaussian', Uhat, R_gauss)));
    k_gauss = d*(d-1)/2;
    AICc_Gauss = -2*ll_gauss + 2*k_gauss + 2*k_gauss*(k_gauss+1)/(n-k_gauss-1);
    
    try
        [R_t, nu_t] = copulafit('t', Uhat);
    catch ME
        warning(ME.identifier, 't Copula fitting failed: %s, using default nu=4', ME.message);
        nu_t = 4;
        R_t = copulafit('Gaussian', Uhat);
    end
    ll_t = sum(log(copulapdf('t', Uhat, R_t, nu_t)));
    k_t = d*(d-1)/2 + 1;
    AICc_t = -2*ll_t + 2*k_t + 2*k_t*(k_t+1)/(n-k_t-1);
    
    if AICc_t < AICc_Gauss
        chosen_type = 't';
        chosen_params = struct('R', R_t, 'nu', nu_t);
    else
        chosen_type = 'Gaussian';
        chosen_params = struct('R', R_gauss);
    end
    
    if strcmp(chosen_type, 't')
        sim_U = copularnd('t', chosen_params.R, chosen_params.nu, Nsim);
    else
        sim_U = copularnd('Gaussian', chosen_params.R, Nsim);
    end
    
    model_info = struct(...
        'R_gauss', R_gauss, 'R_t', R_t, 'nu_t', nu_t,...
        'AICc_Gauss', AICc_Gauss, 'AICc_t', AICc_t,...
        'chosen_type', chosen_type, 'chosen_params', chosen_params...
    );
end

%% ===================== Core Method 2: DTC 3D Simulation (AICc) =====================
function [sim_U, model_info] = DTC_3d_model(Uhat, Nsim)
    [n, d] = size(Uhat);
    if d ~= 3
        error('Input Uhat must be an n x 3 matrix (3D variables), current dimension: n x %d', d);
    end
    if ~(isnumeric(Nsim) && mod(Nsim,1)==0 && Nsim>0)
        error('Nsim must be a positive integer, current value: %s', num2str(Nsim));
    end
    
    corr_mat = corr(Uhat, 'type', 'Kendall');
    abs_corr = abs(corr_mat);
    abs_corr(1:d+1:end) = 0;
    pairs = [[1,2]; [1,3]; [2,3]];
    pairs_list = [];
    for i = 1:size(pairs,1)
        var1 = pairs(i,1); var2 = pairs(i,2);
        pairs_list = [pairs_list; abs_corr(var1,var2), var1, var2];
    end
    pairs_list = sortrows(pairs_list, 1, 'descend');
    top1_pair = pairs_list(1,2:3);
    top2_pair = pairs_list(2,2:3);
    overlap = intersect(top1_pair, top2_pair);
    if isempty(overlap)
        warning('Top two correlated pairs have no overlap, defaulting center node to X2');
        mid_node = 2;
    else
        mid_node = overlap(1);
    end
    other_vars = setdiff(1:d, mid_node);
    copula_pairs = [
        sort([mid_node, other_vars(1)]);
        sort([mid_node, other_vars(2)])
    ];
    
    candidate_types = {'Gaussian', 't', 'Frank', 'Gumbel', 'Clayton'};
    num_pairs = size(copula_pairs,1);
    opt_types = cell(1, num_pairs);
    opt_params = cell(1, num_pairs);
    for k = 1:num_pairs
        pair_idx = copula_pairs(k,:);
        upair = Uhat(:, pair_idx);
        [best_type, best_params] = fit_pairwise_copula(upair, candidate_types);
        opt_types{k} = best_type;
        opt_params{k} = best_params;
    end
    
    % Sampling
    sim_U = zeros(Nsim, d);
    for i = 1:Nsim
        s = rand(1,3);
        u_current = zeros(1,d);
        u_current(mid_node) = s(1);
        
        pair1 = copula_pairs(1,:);
        typ1 = opt_types{1}; params1 = opt_params{1};
        u_cond1 = u_current(mid_node); w1 = s(2);
        pos1 = find(pair1==mid_node,1);
        v_j = hinv_conditional(typ1, params1, u_cond1, w1, pos1, pair1);
        u_current(other_vars(1)) = min(max(v_j, 1e-12), 1-1e-12);
        
        pair2 = copula_pairs(2,:);
        typ2 = opt_types{2}; params2 = opt_params{2};
        u_cond2 = u_current(mid_node); w2 = s(3);
        pos2 = find(pair2==mid_node,1);
        v_k = hinv_conditional(typ2, params2, u_cond2, w2, pos2, pair2);
        u_current(other_vars(2)) = min(max(v_k, 1e-12), 1-1e-12);
        
        sim_U(i,:) = u_current;
    end
    
    model_info = struct(...
        'corr_mat', corr_mat, 'mid_node', mid_node,...
        'copula_pairs', copula_pairs, 'opt_types', {opt_types},...
        'opt_params', {opt_params}, 'top_corr_pairs', pairs_list(1:2,:)...
    );
end

%% ===================== Helper Function 1: Pairwise Copula Fitting (AICc Selection) =====================
function [best_type, best_params] = fit_pairwise_copula(u_pair, candidate_types)
    n = size(u_pair,1);
    best_AICc = Inf; best_type = ''; best_params = [];
    for t = 1:length(candidate_types)
        type = candidate_types{t};
        try
            switch lower(type)
                case 't'
                    [Rhat, nuhat] = copulafit('t', u_pair);
                    ll = sum(log(copulapdf('t', u_pair, Rhat, nuhat)));
                    k = 2;
                    params = struct('R', Rhat, 'nu', nuhat);
                case 'gaussian'
                    Rhat = copulafit('Gaussian', u_pair);
                    ll = sum(log(copulapdf('Gaussian', u_pair, Rhat)));
                    k = 1;
                    params = struct('R', Rhat);
                otherwise
                    theta = copulafit(type, u_pair);
                    ll = sum(log(copulapdf(type, u_pair, theta)));
                    k = 1;
                    params = struct('theta', theta);
            end
            AIC = -2*ll + 2*k;
            AICc = AIC + 2*k*(k+1)/(n-k-1);
            
            if AICc < best_AICc
                best_AICc = AICc; best_type = type; best_params = params;
            end
        catch ME
            warning('Fitting %s Copula failed: %s, skipping', type, ME.message);
        end
    end
    if isempty(best_type)
        warning('All Copula fittings failed, falling back to Gaussian Copula');
        Rhat = copulafit('Gaussian', u_pair);
        best_type = 'Gaussian'; best_params = struct('R', Rhat);
    end
end

%% ===================== Helper Function 2: Inverse Conditional Copula Distribution =====================
function v = hinv_conditional(type, params, u_cond, w, idx_in_pair_cond, pair)
    if idx_in_pair_cond == 1
        switch lower(type)
            case 'gaussian'
                rho = params.R(1,2); 
                v = hinv_gaussian(w, u_cond, rho);
            case 't'
                rho = params.R(1,2);
                v = hinv_t(w, u_cond, rho, params.nu);
            otherwise
                f = @(v) numeric_partial_h(type, params, u_cond, v, 1) - w;
                v = hinv_numeric(f);
        end
    else
        switch lower(type)
            case 'gaussian'
                rho = params.R(1,2);
                v = hinv_gaussian(w, u_cond, rho);
            case 't'
                rho = params.R(1,2);
                v = hinv_t(w, u_cond, rho, params.nu);
            otherwise
                f = @(v) numeric_partial_h(type, params, v, u_cond, 2) - w;
                v = hinv_numeric(f);
        end
    end
end

%% ===================== Other Helper Functions =====================
function u_j = hinv_numeric(f)
    lb = 1e-8; ub = 1-1e-8; x0 = 0.5;
    if f(lb)*f(ub) > 0
        grid = linspace(lb, ub, 9);
        vals = arrayfun(f, grid);
        idx = find(vals(1:end-1).*vals(2:end) < 0, 1);
        if ~isempty(idx)
            lb = grid(idx); ub = grid(idx+1); x0 = (lb+ub)/2;
        else
            [~, closest_idx] = min(abs(vals)); x0 = grid(closest_idx);
        end
    end
    try
        u_j = fzero(f, x0);
    catch
        u_j = bisection(f, lb, ub, 1e-8, 100);
    end
    u_j = min(max(u_j, 1e-12), 1-1e-12);
end

function root = bisection(f, a, b, tol, maxit)
    fa = f(a); fb = f(b);
    if fa*fb > 0
        root = (a+b)/2; warning('Bisection interval has no sign change, returning midpoint'); return;
    end
    for k=1:maxit
        c = (a+b)/2; fc = f(c);
        if abs(fc) < tol || (b-a)/2 < tol
            root = c; return;
        end
        if fa*fc <= 0
            b = c; fb = fc;
        else
            a = c; fa = fc;
        end
    end
    root = (a+b)/2; warning('Bisection reached maximum iterations');
end

function h = numeric_partial_h(type, params, u, v, w_r_t)
    epsu = 1e-6;
    if w_r_t == 1
        u1 = min(max(u+epsu, 0), 1); u0 = min(max(u-epsu, 0), 1);
        uu1 = [u1, v]; uu0 = [u0, v];
    else
        v1 = min(max(v+epsu, 0), 1); v0 = min(max(v-epsu, 0), 1);
        uu1 = [u, v1]; uu0 = [u, v0];
    end
    denom = 2*epsu;
    try
        switch lower(type)
            case 'gaussian'
                C1 = copulacdf('Gaussian', uu1, params.R);
                C0 = copulacdf('Gaussian', uu0, params.R);
            case 't'
                C1 = copulacdf('t', uu1, params.R, params.nu);
                C0 = copulacdf('t', uu0, params.R, params.nu);
            otherwise
                C1 = copulacdf(type, uu1, params.theta);
                C0 = copulacdf(type, uu0, params.theta);
        end
        h = (C1 - C0)/denom;
    catch ME
        warning(ME.identifier, 'Partial derivative calculation failed, adjusting step size: %s', ME.message);
        epsu = 1e-4;
        if w_r_t == 1
            u1 = min(max(u+epsu, 0), 1); u0 = min(max(u-epsu, 0), 1);
            uu1 = [u1, v]; uu0 = [u0, v];
        else
            v1 = min(max(v+epsu, 0), 1); v0 = min(max(v-epsu, 0), 1);
            uu1 = [u, v1]; uu0 = [u, v0];
        end
        denom = 2*epsu;
        switch lower(type)
            case 'gaussian'
                C1 = copulacdf('Gaussian', uu1, params.R);
                C0 = copulacdf('Gaussian', uu0, params.R);
            case 't'
                C1 = copulacdf('t', uu1, params.R, params.nu);
                C0 = copulacdf('t', uu0, params.R, params.nu);
            otherwise
                C1 = copulacdf(type, uu1, params.theta);
                C0 = copulacdf(type, uu0, params.theta);
        end
        h = (C1 - C0)/denom;
    end
end

function v = hinv_gaussian(w, u, rho)
    z_u = norminv(u);
    z_w = norminv(w);
    z_v = rho*z_u + sqrt(max(0,1-rho^2))*z_w;
    v = normcdf(z_v);
end

function v = hinv_t(w, u, rho, nu)
    z_u = tinv(u, nu);
    mean_c = rho*z_u;
    scale_c = sqrt((nu + z_u.^2)./(nu + 1) * (1 - rho^2));
    t_w = tinv(w, nu + 1);
    t_v = mean_c + scale_c.*t_w;
    v = tcdf(t_v, nu);
end

%%%--------------- Existing Helper Functions --------------------%%%
function p = cdf_wrapper(pd, x)
    if isstruct(pd)
        p = pd.cdf(x);
    else
        p = cdf(pd, x);
    end
end

% Pearson3 distribution fitting
function pd = pearson3fit(x)
    m = mean(x); s = std(x); g1 = skewness(x);
    if abs(g1) < 1e-6, g1 = 1e-6; end
    k = 4/(g1^2); theta = s/sqrt(k); a = m - k*theta;
    pd.k = k; pd.theta = theta; pd.a = a;
    pd.pdf = @(xx) pearson3_pdf(xx, k, theta, a);
    pd.cdf = @(xx) pearson3_cdf(xx, k, theta, a);
    pd.icdf = @(u) a + gaminv(u, k, theta);
end

function y = pearson3_pdf(x, k, theta, a)
    z = x - a; y = zeros(size(z)); idx = z > 0;
    y(idx) = gampdf(z(idx), k, theta);
end

function F = pearson3_cdf(x, k, theta, a)
    z = x - a; F = zeros(size(z)); idx = z > 0;
    F(idx) = gamcdf(z(idx), k, theta);
end