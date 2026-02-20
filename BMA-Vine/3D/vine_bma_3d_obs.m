% 3D Observed + Marginal Fitting + Vine Copula + BMA Weighting + Statistical Comparison + Correlation Comparison
clear; clc; close all; rng(1);
%% 1. Observed 3D Data 
% load QZ_Data.mat;
% data = QZ_Data;
load LTZ_Data.mat;
data = LTZ_Data;
n = length(data); d = 3;  % 3D variables
%% 2. Marginal Fitting and Selection
% 6 distribution functions commonly used in hydrology
families = {'Gamma','Lognormal','Pearson3','GEV','Weibull','GPD'};
marg_fits = cell(1,d);
for i = 1:d
    x = data(:, i);
    bestBIC = Inf;
    
    % For simplicity, manually set the threshold for GPD distribution (theta parameter)
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
                    pd = gevfit(x); p = 3;  % GEV distribution parameter estimation
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
    
    % Goodness-of-Fit test (KS test)
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
R_empirical = corrcoef(Uhat);
fprintf('\n--- Original Empirical Distribution Correlation Matrix ---\n');
disp(R_empirical);
%% 4. Marginal Distribution Diagnostics: QQ Plots and Goodness of Fit
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
%% 5. Copula Candidate Families
copula_families = {'Gaussian','t','Clayton','Gumbel','Frank'};
%% 6. Vine Copula Structure Selection (Three unique structures)
unique_vine_orders = {
    [1, 2, 3];
    [2, 1, 3];
    [3, 1, 2]
};
fprintf('Using %d unique Vine structures for 3D case.\n', length(unique_vine_orders));
vine_models = cell(1, length(unique_vine_orders));
bestVine.BIC = Inf;
for i = 1:length(unique_vine_orders)
    ord = unique_vine_orders{i};
    try
        % Use revised Vine fitting function (no longer using numeric handles)
        model = fit_three_dim_vine(Uhat, ord, copula_families);
        vine_models{i} = model;
        if model.BIC < bestVine.BIC
            bestVine = model;
        end
    catch ME
        fprintf('Vine fit failed for order %s: %s\n', mat2str(ord), ME.message);
        vine_models{i} = [];
    end
end
%% Display preferred copula functions for Tree 1 and Tree 2 of each vine structure
fprintf('\n=== Detailed Copula Information for Each Vine Structure ===\n');
valid_vine_models = vine_models(~cellfun('isempty',vine_models));
valid_model_idx = 0;
for i = 1:length(vine_models)
    model = vine_models{i};
    if isempty(model), continue; end
    valid_model_idx = valid_model_idx + 1;
    ord = model.order;
    tree1_copulas = model.Trees{1}.fams;
    tree2_copulas = model.Trees{2}.fams;
    
    fprintf('\nVine Structure %d: Order = %s, BIC = %.3f\n', ...
        valid_model_idx, mat2str(ord), model.BIC);
    fprintf('  Tree 1 Copulas:\n');
    fprintf('    - Variable pair (%d, %d): %s\n', ord(1), ord(2), tree1_copulas{1});
    fprintf('    - Variable pair (%d, %d): %s\n', ord(1), ord(3), tree1_copulas{2});
    fprintf('  Tree 2 Copulas:\n');
    fprintf('    - Conditional pair (%d, %d)|%d: %s\n', ...
        ord(2), ord(3), ord(1), tree2_copulas{1});
end
%% Output best results
fprintf('\n=== Best Vine structure (from %d unique vines) ===\n', valid_model_idx);
if isfield(bestVine,'order')
    fprintf('Order: %s, BIC: %.3f\n', mat2str(bestVine.order), bestVine.BIC);
    fprintf('Best Vine Copula Details:\n');
    fprintf('  Tree 1 Copulas:\n');
    fprintf('    - Variable pair (%d, %d): %s\n', ...
        bestVine.order(1), bestVine.order(2), bestVine.Trees{1}.fams{1});
    fprintf('    - Variable pair (%d, %d): %s\n', ...
        bestVine.order(1), bestVine.order(3), bestVine.Trees{1}.fams{2});
    fprintf('  Tree 2 Copulas:\n');
    fprintf('    - Conditional pair (%d, %d)|%d: %s\n', ...
        bestVine.order(2), bestVine.order(3), bestVine.order(1), bestVine.Trees{2}.fams{1});
else
    fprintf('No Vine model successfully fitted.\n');
end
%% 7. BMA Weight Calculation (using valid models)
num_models = length(valid_vine_models);
if num_models == 0
    error('No Vine models were successfully fitted for BMA.');
end
BICs = zeros(1, num_models);
for k = 1:num_models
    BICs(k) = valid_vine_models{k}.BIC;
end
logL_rel = -0.5 * BICs;
max_logL_rel = max(logL_rel);
w_numerator = exp(logL_rel - max_logL_rel);
w = w_numerator / sum(w_numerator);
fprintf('\n=== BMA Weights for Each Vine Structure ===\n');
for k = 1:num_models
    ord = valid_vine_models{k}.order;
    fprintf('Vine Structure (Order %s): BIC = %.3f, Weight = %.5f\n', ...
        mat2str(ord), BICs(k), w(k));
end
fprintf('Total BIC Range: [%.3f, %.3f]\n', min(BICs), max(BICs));
fprintf('Sum of weights: %.5f\n', sum(w));
%% 8. Simulation
Nsim = 1000;
simBMA = [];
for k = 1:num_models
    Nk = round(Nsim * w(k));
    if Nk > 0
        model = valid_vine_models{k};
        simk = simulate_three_dim_vine(Nk, model);
        simBMA = [simBMA; simk]; %#ok<AGROW>
    end
end
Nsim_actual = size(simBMA, 1);
fprintf('Note: Total simulated samples adjusted to Nsim=%d due to rounding.\n', Nsim_actual);
fprintf('\n--- Simulated CDF Correlation Matrix (before conversion) ---\n');
R_sim_cdf = corrcoef(simBMA);
disp(R_sim_cdf);
% Convert back to original data space
simBMA_data = zeros(size(simBMA));
for i = 1:d
    pd = marg_fits{i}.pd;
    fam = marg_fits{i}.family;
    if strcmp(fam, 'GEV')
        simBMA_data(:, i) = gevinv(simBMA(:, i), pd(1), pd(2), pd(3));
    elseif strcmp(fam, 'GPD')
        simBMA_data(:, i) = gpdinv(simBMA(:, i), pd(1), pd(2), pd(3));
    elseif isstruct(pd)
        simBMA_data(:, i) = pd.icdf(simBMA(:, i));
    else
        simBMA_data(:, i) = icdf(pd, simBMA(:, i));
    end
end
simBMA = simBMA_data;
%% 9. Compare Statistics 
fprintf('\n--- Comparing Means and Standard Deviations ---\n');
for j = 1:d
    fprintf('Var%d mean: orig=%.3f sim=%.3f | std: orig=%.3f sim=%.3f\n', ...
        j, mean(data(:, j)), mean(simBMA(:, j)), std(data(:, j)), std(simBMA(:, j)));
end
%% 10. Compare Correlation Structures 
R_orig = corrcoef(data);
R_sim = corrcoef(simBMA);
fprintf('\n--- Comparing Correlation Structures ---\n');
fprintf('Original data sample correlation matrix:\n');
disp(R_orig);
fprintf('BMA simulated data sample correlation matrix:\n');
disp(R_sim);
%% 11. Scatter Comparison 
pairs = nchoosek(1:d, 2);
figure('Position', [200 200 1200 400]);
alpha_val = 0.3;
original_color = [0 0.447 0.741];
simulated_color = [0.85 0.333 0.1];
for k = 1:size(pairs, 1)
    subplot(1, 3, k);
    i = pairs(k, 1); j = pairs(k, 2);
    scatter(simBMA(:, i), simBMA(:, j), 30, simulated_color, 'filled', ...
        'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', alpha_val);
    hold on;
    scatter(data(:, i), data(:, j), 30, original_color, 'filled', ...
        'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', alpha_val);
    
    xlabel(sprintf('X%d', i), 'FontWeight', 'bold'); 
    ylabel(sprintf('X%d', j), 'FontWeight', 'bold');
    title(sprintf('X%d vs X%d', i, j), 'FontWeight', 'bold');
    grid on; box on;
    if k == 1
        legend({'Simulated','Original'}, 'Location', 'best', 'Box', 'off');
    end
end
sgtitle('Scatter Plots: Original vs BMA Simulated', 'FontSize', 14, 'FontWeight', 'bold');
set(groot, 'DefaultLineLineWidth', 0.5);
set(groot, 'DefaultAxesFontName', 'Helvetica');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Helvetica');
%% =================== Function Definition Section ===================
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
%% --------- Vine Related Functions ---------
function model = fit_three_dim_vine(U, order, fams)
    n = size(U,1);
    Uord = U(:, order);
    [f12, p12, logL12, k12] = fit_pair(Uord(:,1), Uord(:,2), fams);
    [f13, p13, logL13, k13] = fit_pair(Uord(:,1), Uord(:,3), fams);
    total_logL = logL12 + logL13;
    total_k = k12 + k13;
    U2g1 = cond_cdf(f12, p12, Uord(:,1), Uord(:,2));
    U3g1 = cond_cdf(f13, p13, Uord(:,1), Uord(:,3));
    [f23g1, p23g1, logL23, k23] = fit_pair(U2g1, U3g1, fams);
    total_logL = total_logL + logL23;
    total_k = total_k + k23;
    model.BIC = -2 * total_logL + total_k * log(n);
    model.order = order;
    model.Trees{1}.fams = {f12, f13};
    model.Trees{1}.params = {p12, p13};
    model.Trees{2}.fams = {f23g1};
    model.Trees{2}.params = {p23g1};
end
function [fam, param, logL, k] = fit_pair(u, v, fams)
    bestL = -Inf; bestFam=''; bestParam = []; bestK = 0;
    for ii = 1:length(fams)
        current_fam = fams{ii};
        try
            switch lower(current_fam)
                case 'gaussian'
                    rho = copulafit('Gaussian', [u,v]);
                    pdfv = copulapdf('Gaussian', [u,v], rho);
                    param.type = 'Gaussian'; param.rho = rho; k = 1;
                case 't'
                    [R, nu] = copulafit('t', [u,v]);
                    pdfv = copulapdf('t', [u,v], R, nu);
                    param.type = 't'; param.R = R; param.nu = nu; k = 2;
                otherwise
                    theta = copulafit(current_fam, [u,v]);
                    pdfv = copulapdf(current_fam, [u,v], theta);
                    param.type = current_fam; param.theta = theta; k = 1;
            end
            logLv = sum(log(max(pdfv,realmin)));
            if logLv > bestL
                bestL = logLv; bestFam = current_fam; bestParam = param; bestK = k;
            end
        catch ME
            fprintf('⚠ pair fit failed for %s: %s\n', current_fam, ME.message);
            continue;
        end
    end
    fam = bestFam; param = bestParam; logL = bestL; k = bestK;
end
function H = cond_cdf(fam, param, u1, u2)
    n = length(u1); H = zeros(n,1);
    eps_base = 1e-6;
    for i = 1:n
        uu1 = u1(i); uu2 = u2(i);
        epsv = max(eps_base, min(1e-3, 1e-4*max(1,abs(uu1))));
        a1 = min(max(uu1 + epsv, epsv), 1-epsv);
        a2 = min(max(uu1 - epsv, epsv), 1-epsv);
        try
            switch param.type
                case 't'
                    C1 = copulacdf('t',[a1, uu2], param.R, param.nu);
                    C2 = copulacdf('t',[a2, uu2], param.R, param.nu);
                case 'Gaussian'
                    C1 = copulacdf('Gaussian',[a1, uu2], param.rho);
                    C2 = copulacdf('Gaussian',[a2, uu2], param.rho);
                otherwise
                    C1 = copulacdf(param.type, [a1, uu2], param.theta);
                    C2 = copulacdf(param.type, [a2, uu2], param.theta);
            end
        catch
            C1 = a1 * uu2; C2 = a2 * uu2;
        end
        H(i) = (C1 - C2) / (a1 - a2);
        H(i) = min(max(H(i),0),1);
    end
end
function val = cond_cdf_single(fam, param, u1, u2)
    epsv = 1e-6;
    a1 = min(max(u1 + epsv, epsv), 1-epsv);
    a2 = min(max(u1 - epsv, epsv), 1-epsv);
    try
        switch param.type
            case 't'
                C1 = copulacdf('t', [a1, u2], param.R, param.nu);
                C2 = copulacdf('t', [a2, u2], param.R, param.nu);
            case 'Gaussian'
                C1 = copulacdf('Gaussian', [a1, u2], param.rho);
                C2 = copulacdf('Gaussian', [a2, u2], param.rho);
            otherwise
                C1 = copulacdf(param.type, [a1, u2], param.theta);
                C2 = copulacdf(param.type, [a2, u2], param.theta);
        end
    catch
        C1 = a1 * u2; C2 = a2 * u2;
    end
    val = (C1 - C2) / (a1 - a2);
    val = min(max(val,0),1);
end
function u_sol = inv_cond_cdf(fam, param, target, known_u, invert_for)
    N = length(target);
    u_sol = zeros(N,1);
    tol = 1e-8; maxit = 60;
    for i = 1:N
        t = min(max(target(i),0),1);
        known = min(max(known_u(i), eps), 1-eps);
        lb = eps; ub = 1-eps;
        if strcmpi(invert_for,'second')
            g = @(x) cond_cdf_single(fam, param, known, x) - t;
        else
            g = @(x) cond_cdf_single(fam, param, x, known) - t;
        end
        g_lb = g(lb); g_ub = g(ub);
        if g_lb == 0, u_sol(i) = lb; continue; end
        if g_ub == 0, u_sol(i) = ub; continue; end
        if g_lb * g_ub > 0
            grid = linspace(lb, ub, 201);
            if strcmpi(invert_for,'second')
                vals = arrayfun(@(x) cond_cdf_single(fam,param,known,x), grid);
            else
                vals = arrayfun(@(x) cond_cdf_single(fam,param,x,known), grid);
            end
            [~, idx] = min(abs(vals - t));
            u_sol(i) = grid(idx); continue;
        end
        a = lb; b = ub; fa = g_lb; fb = g_ub; c = a;
        for it = 1:maxit
            c = 0.5*(a+b); fc = g(c);
            if abs(fc) < tol || (b-a)/2 < tol, break; end
            if fa*fc <= 0
                b = c; fb = fc;
            else
                a = c; fa = fc;
            end
        end
        u_sol(i) = c;
    end
end
function U_sim = simulate_three_dim_vine(N, model)
    Trees = model.Trees;
    order = model.order(:)';
    d = length(order);
    S = rand(N,d);
    U_sim_ordered = zeros(N,d);
    f12 = Trees{1}.fams{1}; p12 = Trees{1}.params{1};
    f13 = Trees{1}.fams{2}; p13 = Trees{1}.params{2};
    f23g1 = Trees{2}.fams{1}; p23g1 = Trees{2}.params{1};
    u1 = S(:,1); 
    U_sim_ordered(:,1) = u1;
    u2 = inv_cond_cdf(f12, p12, S(:,2), u1, 'second');
    U_sim_ordered(:,2) = u2;
    u2g1 = cond_cdf(f12, p12, u1, u2);
    u3g1 = inv_cond_cdf(f23g1, p23g1, S(:,3), u2g1, 'second');
    u3 = inv_cond_cdf(f13, p13, u3g1, u1, 'second');
    U_sim_ordered(:,3) = u3;
    U_sim = zeros(N,d);
    U_sim(:, order) = U_sim_ordered;
end