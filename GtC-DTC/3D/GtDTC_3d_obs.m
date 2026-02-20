%% 3D Gaussian/t Copula and DTC Sampling methods
clear; clc; close all;rng(1);
%% ===================== Top-Level Execution Logic =====================
%% 1. Observed 3D Data 
load QZ_Data.mat;
data = QZ_Data;
% load LTZ_Data.mat;
% data = LTZ_Data;
n = length(data); d = 3;  % 3D variables
%% 2. Marginal Fitting and Selection
% 6 common distribution functions in hydrology
families = {'Gamma','Lognormal','Pearson3','GEV','Weibull','GPD'};
marg_fits = cell(1,d);
for i = 1:d
    x = data(:, i);
    bestBIC = Inf;
    
    % Manually set GPD threshold (theta parameter) to simplify calculation
    theta_gpd = min(x) - 1; 
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
                case 'GPD'
                    % GPD fitting, parameters are [k, sigma, theta]
                    [param_gpd, p_gpd] = gpdfit(x, theta_gpd); 
                    pd = param_gpd; 
                    p = p_gpd; % GPD has 2 free parameters (k, sigma)
            end
            
            % Calculate log-likelihood and BIC
            if strcmp(fam, 'GEV')
                pdfv = gevpdf(x, pd(1), pd(2), pd(3));
            elseif strcmp(fam, 'GPD')
                pdfv = gpdpdf(x, pd(1), pd(2), pd(3));
            elseif isstruct(pd)
                pdfv = pd.pdf(x);
            else
                pdfv = pdf(pd, x);
            end
            pdfv(pdfv <= 0) = realmin;
            logL = sum(log(pdfv));
            BIC = -2 * logL + p * log(n);
            
            % Update best model
            if BIC < bestBIC
                bestBIC = BIC; 
                bestFit = pd; 
                bestFam = fam;
                if strcmp(fam, 'GPD')
                    bestFit_theta_gpd = theta_gpd;
                else
                    bestFit_theta_gpd = NaN;
                end
            end
        catch
            continue;
        end
    end
    marg_fits{i}.pd = bestFit;
    marg_fits{i}.family = bestFam;
    if strcmp(bestFam, 'GPD')
        marg_fits{i}.theta_gpd = bestFit_theta_gpd;
    else
        marg_fits{i}.theta_gpd = NaN;
    end
    
    fprintf('Var%d: best marginal = %s\n', i, bestFam);
    
    % Goodness-of-fit test (KS test)
    fam = bestFam;
    pd = bestFit;
    if strcmp(fam, 'GEV')
        Fx = gevcdf(x, pd(1), pd(2), pd(3));
    elseif strcmp(fam, 'GPD')
        Fx = gpdcdf(x, pd(1), pd(2), pd(3));
    elseif isstruct(pd)
        Fx = pd.cdf(x);
    else
        Fx = cdf(pd, x);
    end
    
    try
        [~, ks_p] = kstest(x, [x, Fx]);
    catch
        ks_p = NaN;
    end
    marg_fits{i}.ks_p = ks_p;
    
    fprintf('Var%d Goodness-of-Fit Test:\n', i);
    fprintf('  KS test p-value:    %.4f\n', ks_p);
    fprintf('  (p-value > 0.05 indicates good fit)\n\n');
end
%% 3. Get Pseudo-Observations (Using Empirical Distribution)
Uhat = zeros(n,d);
for i=1:d
    x = data(:,i);
    [~, idx] = sort(x);
    ranks = zeros(n, 1);
    ranks(idx) = (1:n)';
    Uhat(:,i) = ranks / (n + 1);
end
Uhat = max(realmin, min(1-realmin, Uhat));
Rtrue = corrcoef(Uhat);
fprintf('\n--- Original Empirical Distribution Correlation Matrix ---\n');
disp(Rtrue);
sample_corr = corr(data);
%% 4. Marginal Distribution Diagnostics: QQ Plots and Goodness-of-Fit
set(groot, 'DefaultLineLineWidth', 1.2);
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Times New Roman');
figure('Name', 'Marginal Distribution QQ Plots', 'Position', [100 100 1200 400]);
for i = 1:d
    subplot(1, 3, i);
    x = data(:, i);
    pd = marg_fits{i}.pd;
    fam = marg_fits{i}.family;
    
    nobs = length(x);
    p = (1:nobs)' / (nobs + 1);
    if strcmp(fam, 'GEV')
        q_theo = gevinv(p, pd(1), pd(2), pd(3));
    elseif strcmp(fam, 'GPD')
        q_theo = gpdinv(p, pd(1), pd(2), pd(3));
    elseif isstruct(pd)
        q_theo = pd.icdf(p);
    else
        q_theo = icdf(pd, p);
    end
    q_emp = sort(x);
    
    plot(q_theo, q_emp, 'o', 'MarkerSize', 4, ...
        'MarkerEdgeColor', [0.2 0.2 0.8], 'MarkerFaceColor', [0.7 0.7 1]);
    hold on;
    min_val = min([q_theo; q_emp]);
    max_val = max([q_theo; q_emp]);
    line([min_val, max_val], [min_val, max_val], ...
        'Color', [0.8 0 0], 'LineStyle', '--');
    xlabel(sprintf('Theoretical Quantiles (%s)', fam), 'FontWeight', 'bold');
    ylabel('Sample Quantiles', 'FontWeight', 'bold');
    title(sprintf('Variable %d', i), 'FontWeight', 'bold');
    grid on; box on;
    
    text_str = sprintf('KS p: %.3f', marg_fits{i}.ks_p);
    text(0.05, 0.95, text_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'FontSize', 8, ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');
end
sgtitle('Marginal Distribution QQ Plots with KS Test', ...
    'FontSize', 14, 'FontWeight', 'bold');

% 2. Method 1: Gaussian/t Copula Simulation
Nsim = 1000;
fprintf('=== Running Gaussian/t Copula 3D Simulation ===\n');
[sim_U_gauss_t, model_info_gauss_t] = gaussian_t_copula_3d_model(Uhat, Nsim);
fprintf('Chosen Copula Type: %s\n', model_info_gauss_t.chosen_type);
fprintf('Gaussian Copula BIC: %.2f | t Copula BIC: %.2f\n\n', ...
    model_info_gauss_t.BIC_Gauss, model_info_gauss_t.BIC_t);
    
    % Calculate and display correlation matrix comparison for Method 1
    fprintf('=== Method 1 (Gaussian/t) Correlation Matrix Comparison ===\n');
    fprintf('True Correlation Matrix:\n');
    disp(Rtrue);
    fprintf('\nSimulated Data Correlation Matrix:\n');
    sim_corr_gauss_t = corr(sim_U_gauss_t);
    disp(sim_corr_gauss_t);
    fprintf('\nCorrelation Matrix Error (Frobenius norm): %.6f\n\n', norm(Rtrue - sim_corr_gauss_t, 'fro'));

% 3. Method 2: DTC Sampling Simulation
fprintf('=== Running DTC 3D Simulation ===\n');
[sim_U_gibbs, model_info_gibbs] = copula_gibbs_3d_model(Uhat, Nsim);
fprintf('Central Node Variable (X1/X2/X3): %d\n', model_info_gibbs.mid_node);
% ====== Added: Display selected copula pairs ======
fprintf('Selected Copula Pairs (sorted):\n');
disp(model_info_gibbs.copula_pairs);
opt_types_local = model_info_gibbs.opt_types;
% Use placeholder if length < 2
if length(opt_types_local) < 2
    opt1 = 'NA'; opt2 = 'NA';
else
    opt1 = opt_types_local{1};
    opt2 = opt_types_local{2};
end
fprintf('Optimal Copula for Pair 1: %s | Optimal Copula for Pair 2: %s\n\n', opt1, opt2);

% Calculate and display correlation matrix comparison for Method 2
fprintf('=== Method 2 (DTC) Correlation Matrix Comparison ===\n');
fprintf('True Correlation Matrix:\n');
disp(Rtrue);
fprintf('\nSimulated Data Correlation Matrix:\n');
sim_corr_gibbs = corr(sim_U_gibbs);
disp(sim_corr_gibbs);
fprintf('\nCorrelation Matrix Error (Frobenius norm): %.6f\n\n', norm(Rtrue - sim_corr_gibbs, 'fro'));

% Compare both methods
fprintf('=== Method Comparison ===\n');
fprintf('Sample Correlation Matrix (from original data):\n');
disp(sample_corr);
fprintf('\nMethod 1 (Gaussian/t) Correlation Error: %.6f\n', norm(Rtrue - sim_corr_gauss_t, 'fro'));
fprintf('Method 2 (DTC) Correlation Error: %.6f\n', norm(Rtrue - sim_corr_gibbs, 'fro'));
if norm(Rtrue - sim_corr_gauss_t, 'fro') < norm(Rtrue - sim_corr_gibbs, 'fro')
    fprintf('\nMethod 1 (Gaussian/t) better preserves the true correlation structure.\n');
else
    fprintf('\nMethod 2 (DTC) better preserves the true correlation structure.\n');
end

% ===================== Convert Simulated Data to Original Scale =====================
fprintf('\n=== Converting Simulated Uniform Data to Original Scale ===\n');
% Transform simulated uniform data back to original scale using the fitted marginal distributions
% We'll convert all Nsim samples
% Method 1: Gaussian/t Copula simulation back to original scale
sim_data_gauss_t = zeros(Nsim, 3);
% Convert back to original data space
for i = 1:d
    pd = marg_fits{i}.pd;
    fam = marg_fits{i}.family;
    if strcmp(fam, 'GEV')
        sim_data_gauss_t(:, i) = gevinv(sim_U_gauss_t(:, i), pd(1), pd(2), pd(3));
    elseif strcmp(fam, 'GPD')
        sim_data_gauss_t(:, i) = gpdinv(sim_U_gauss_t(:, i), pd(1), pd(2), pd(3));
    elseif isstruct(pd)
        sim_data_gauss_t(:, i) = pd.icdf(sim_U_gauss_t(:, i));
    else
        sim_data_gauss_t(:, i) = icdf(pd, sim_U_gauss_t(:, i));
    end
end

% Method 2: DTC simulation back to original scale
sim_data_gibbs = zeros(size(sim_U_gibbs));
for i = 1:d
    pd = marg_fits{i}.pd;
    fam = marg_fits{i}.family;
    if strcmp(fam, 'GEV')
        sim_data_gibbs(:, i) = gevinv(sim_U_gibbs(:, i), pd(1), pd(2), pd(3));
    elseif strcmp(fam, 'GPD')
        sim_data_gibbs(:, i) = gpdinv(sim_U_gibbs(:, i), pd(1), pd(2), pd(3));
    elseif isstruct(pd)
        sim_data_gibbs(:, i) = pd.icdf(sim_U_gibbs(:, i));
    else
        sim_data_gibbs(:, i) = icdf(pd, sim_U_gibbs(:, i));
    end
end

% Display summary statistics of the simulated data in original scale
fprintf('\n=== Summary Statistics of Simulated Data (Original Scale) ===\n');
% Variable names
var_names = {'X1', 'X2', 'X3'};
% Display statistics for each variable and method
for var_idx = 1:3
    fprintf('\n%s Statistics:\n', var_names{var_idx});
    fprintf('  Original Data - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n', ...
        mean(data(:, var_idx)), std(data(:, var_idx)), min(data(:, var_idx)), max(data(:, var_idx)));
    fprintf('  Method 1 (t Copula) - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n', ...
        mean(sim_data_gauss_t(:, var_idx)), std(sim_data_gauss_t(:, var_idx)), min(sim_data_gauss_t(:, var_idx)), max(sim_data_gauss_t(:, var_idx)));
    fprintf('  Method 2 (DTC) - Mean: %.4f, Std: %.4f, Min: %.4f, Max: %.4f\n', ...
        mean(sim_data_gibbs(:, var_idx)), std(sim_data_gibbs(:, var_idx)), min(sim_data_gibbs(:, var_idx)), max(sim_data_gibbs(:, var_idx)));
end

% Display correlation matrices for simulated data in original scale
fprintf('\n=== Correlation Matrices of Simulated Data (Original Scale) ===\n');
corr_original = corr(data);
fprintf('Original Data Correlation Matrix:\n');
disp(corr_original);
corr_method1 = corr(sim_data_gauss_t);
fprintf('\nMethod 1 (t Copula) Correlation Matrix:\n');
disp(corr_method1);
corr_method2 = corr(sim_data_gibbs);
fprintf('\nMethod 2 (DTC) Correlation Matrix:\n');
disp(corr_method2);

% Calculate error for correlation matrices in original scale
error_corr_method1 = norm(corr_original - corr_method1, 'fro');
error_corr_method2 = norm(corr_original - corr_method2, 'fro');
fprintf('\nCorrelation Matrix Error (Frobenius norm) in Original Scale:\n');
fprintf('Method 1 (t Copula): %.6f\n', error_corr_method1);
fprintf('Method 2 (DTC): %.6f\n', error_corr_method2);

% Calculate distribution fitting accuracy 
fprintf('\n=== Distribution Fitting Accuracy ===\n');
for var_idx = 1:3
    % Calculate Kolmogorov-Smirnov test between original and simulated data
    [h1, p1] = kstest2(data(:, var_idx), sim_data_gauss_t(:, var_idx));
    [h2, p2] = kstest2(data(:, var_idx), sim_data_gibbs(:, var_idx));
    
    fprintf('\n%s Distribution Fit Test:\n', var_names{var_idx});
    fprintf('  Method 1 (t Copula) - KS Test p-value: %.4f (Null hypothesis: Same distribution)\n', p1);
    fprintf('  Method 2 (DTC) - KS Test p-value: %.4f (Null hypothesis: Same distribution)\n', p2);
    
    if p1 > 0.05
        fprintf('  Method 1 successfully preserves the marginal distribution (p > 0.05)\n');
    else
        fprintf('  Method 1 fails to preserve the marginal distribution (p <= 0.05)\n');
    end
    
    if p2 > 0.05
        fprintf('  Method 2 successfully preserves the marginal distribution (p > 0.05)\n');
    else
        fprintf('  Method 2 fails to preserve the marginal distribution (p <= 0.05)\n');
    end
end

%% ===================== Core Method 1: Gaussian/t Copula 3D Simulation =====================
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
    BIC_Gauss = -2*ll_gauss + k_gauss*log(n);
    try
        [R_t, nu_t] = copulafit('t', Uhat);
    catch ME
        warning(ME.identifier, 't Copula fitting failed: %s, using default nu=4', ME.message);
        nu_t = 4;
        R_t = copulafit('Gaussian', Uhat);
    end
    ll_t = sum(log(copulapdf('t', Uhat, R_t, nu_t)));
    k_t = d*(d-1)/2 + 1;
    BIC_t = -2*ll_t + k_t*log(n);
    if BIC_t < BIC_Gauss
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
        'BIC_Gauss', BIC_Gauss, 'BIC_t', BIC_t,...
        'chosen_type', chosen_type, 'chosen_params', chosen_params...
    );
end

%% ===================== Core Method 2: DTC 3D Simulation =====================
function [sim_U, model_info] = copula_gibbs_3d_model(Uhat, Nsim)
    [n, d] = size(Uhat);
    if d ~= 3
        error('Input Uhat must be an n x 3 matrix (3D variables), current dimension: n x %d', d);
    end
    if ~(isnumeric(Nsim) && mod(Nsim,1)==0 && Nsim>0)
        error('Nsim must be a positive integer, current value: %s', num2str(Nsim));
    end
    corr_mat = corr(Uhat, 'type', 'Pearson');
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
        upair = Uhat(:, pair_idx);    % Column order matches copula_pairs (sorted)
        [best_type, best_params] = fit_pairwise_copula(upair, candidate_types);
        opt_types{k} = best_type;
        opt_params{k} = best_params;
    end
    % DTC Sampling
    sim_U = zeros(Nsim, d);
    for i = 1:Nsim
        s = rand(1,3);
        u_current = zeros(1,d);
        % Central node
        u_current(mid_node) = s(1);
        % First non-central node
        pair1 = copula_pairs(1,:);
        typ1 = opt_types{1}; params1 = opt_params{1};
        u_cond1 = u_current(mid_node); w1 = s(2);
        pos1 = find(pair1==mid_node,1);
        v_j = hinv_conditional(typ1, params1, u_cond1, w1, pos1, pair1);
        u_current(other_vars(1)) = min(max(v_j, 1e-12), 1-1e-12);
        % Second non-central node
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

%% ===================== Helper Function 1: Pairwise Copula Fitting (BIC Selection) =====================
function [best_type, best_params] = fit_pairwise_copula(u_pair, candidate_types)
    n = size(u_pair,1);
    best_BIC = Inf; best_type = ''; best_params = [];
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
            BIC = -2*ll + k*log(n);
            if BIC < best_BIC
                best_BIC = BIC; best_type = type; best_params = params;
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
    % idx_in_pair_cond: 1 or 2, indicates position of the conditioning variable in pair
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

%% ===================== Additional Helper Functions =====================
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

% GPD distribution fitting function (k, sigma, theta)
function [param_gpd, p_gpd] = gpdfit(x, theta)
    x_exceed = x(x > theta);
    if isempty(x_exceed)
        error('No data exceeds the GPD threshold.');
    end
    param_hat = gpfit(x_exceed - theta); 
    k_hat = param_hat(1);
    sigma_hat = param_hat(2);
    param_gpd = [k_hat, sigma_hat, theta]; 
    p_gpd = 2;
end

function y = gpdpdf(x, k, sigma, theta)
    x_exceed = x - theta;
    y = zeros(size(x));
    idx = x_exceed > 0;
    if k == 0
        y(idx) = (1/sigma) * exp(-x_exceed(idx) / sigma);
    else
        factor = 1 + k * x_exceed(idx) / sigma;
        valid_idx = idx & (factor > 0);
        y(valid_idx) = (1/sigma) * (factor(valid_idx)).^(-1/k - 1);
    end
end

function F = gpdcdf(x, k, sigma, theta)
    x_exceed = x - theta;
    F = zeros(size(x));
    idx = x_exceed > 0;
    if k == 0
        F(idx) = 1 - exp(-x_exceed(idx) / sigma);
    else
        factor = 1 + k * x_exceed(idx) / sigma;
        valid_idx = idx & (factor > 0);
        F(valid_idx) = 1 - (factor(valid_idx)).^(-1/k);
    end
    F = max(0, min(1, F));
end

function x = gpdinv(p, k, sigma, theta)
    x = zeros(size(p));
    idx = p > 0 & p < 1;
    if k == 0
        x(idx) = theta - sigma * log(1 - p(idx));
    else
        x(idx) = theta + (sigma / k) * ((1 - p(idx)).^(-k) - 1);
    end
end

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