%% GtG_4D_obs.m
% 4D Observed Data + Marginal Fitting + Two Copula Simulation Methods + Statistical & Correlation Comparison

clear; clc; close all; rng(1);

%% 1. Input Data
% load ('Storm_NL_4D.mat');
% data = storm_NL_4D;
load preci_yangzhou.mat;
data = preci_yangzhou;
n = length(data);
d = 4;

%% 2. Marginal Fitting and Selection
families = {'Gamma','Lognormal','Pearson3','GEV','Weibull','GPD'};
for i=1:d
    x = data(:,i);
    bestBIC = Inf;
    bestFit_theta_gpd = NaN;  % Store GPD threshold
    
    % Set threshold for GPD
    theta_gpd = min(x) - 1; 

    for f = families
        fam = f{1};
        try
            switch fam
                case 'Gamma'
                    pd = fitdist(x,'Gamma'); p=2;
                case 'Lognormal'
                    pd = fitdist(x,'Lognormal'); p=2;
                case 'Weibull'
                    pd = fitdist(x,'Weibull'); p=2;
                case 'Pearson3'
                    pd = pearson3fit(x); p=3;
                case 'GEV'
                    pd = gevfit(x); p=3;
                case 'GPD'
                    [param_gpd, p_gpd] = gpdfit(x, theta_gpd); 
                    pd = param_gpd; 
                    p = p_gpd;
            end
            
            % Calculate log-likelihood and BIC
            if strcmp(fam, 'GEV')
                pdfv = gevpdf(x, pd(1), pd(2), pd(3));
            elseif strcmp(fam, 'GPD')
                pdfv = gpdpdf(x, pd(1), pd(2), pd(3));
            elseif isstruct(pd)
                pdfv = pd.pdf(x);
            else
                pdfv = pdf(pd,x);
            end
            pdfv(pdfv<=0) = realmin;
            logL = sum(log(pdfv));
            BIC = -2*logL + p*log(n);
            
            % Update best model
            if BIC < bestBIC
                bestBIC = BIC; 
                bestFit = pd; 
                bestFam = fam;
                if strcmp(fam, 'GPD')
                    bestFit_theta_gpd = theta_gpd;
                end
            end
        catch
            continue;
        end
    end
    marg_fits{i}.pd = bestFit;
    marg_fits{i}.family = bestFam;
    marg_fits{i}.theta_gpd = bestFit_theta_gpd;
    fprintf('Var%d: best marginal = %s\n',i,bestFam);
end


%% 3. Get Pseudo-Observations (Using Empirical Distribution)
Uhat = zeros(n,d);
for i=1:d
    x = data(:,i);
    
    % Calculate ranks for empirical CDF using sort
    [~, idx] = sort(x);
    ranks = zeros(n, 1);
    ranks(idx) = (1:n)';
    % Use i/(n+1) formula for empirical CDF
    Uhat(:,i) = ranks / (n + 1);
end
Uhat = max(realmin, min(1-realmin, Uhat));
%Calculate correlation matrix of original empirical distributions (pseudo-observations)
R_empirical = corrcoef(Uhat);
fprintf('\n--- Original Empirical Distribution Correlation Matrix ---\n');
disp(R_empirical);

%% 4. Marginal Distribution Diagnosis: QQ Plot
set(groot, 'DefaultLineLineWidth', 1.2);
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Times New Roman');

figure('Name', 'Marginal Distribution QQ Plots', 'Position', [100 100 800 800]);
for i = 1:d
    subplot(2, 2, i);
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
    plot(q_theo, q_emp, 'o', 'MarkerSize', 4, 'MarkerEdgeColor', [0.2 0.2 0.8], 'MarkerFaceColor', [0.7 0.7 1]);
    hold on;
    min_val = min([q_theo; q_emp]);
    max_val = max([q_theo; q_emp]);
    line([min_val, max_val], [min_val, max_val], 'Color', [0.8 0 0], 'LineStyle', '--');
    xlabel(sprintf('Theoretical Quantiles (%s)', fam), 'FontWeight', 'bold');
    ylabel('Sample Quantiles', 'FontWeight', 'bold');
    title(sprintf('Variable %d', i), 'FontWeight', 'bold');
    grid on;
    box on;
end
sgtitle('Marginal Distribution QQ Plots', 'FontSize', 14, 'FontWeight', 'bold');


%% 5. Select Simulation Method
fprintf('\nPlease select a simulation method:\n');
fprintf('1 - Method 1 (Gaussian/t Copula)\n');
fprintf('2 - Method 2 (DTC sampling)\n');
method_choice = input('Enter 1 or 2: ');
while ~ismember(method_choice, [1,2])
    fprintf('Invalid input, please select again!\n');
    method_choice = input('Enter 1 or 2: ');
end


%% 6. Model Construction and Random Simulation
Nsim = 1000;  % Number of simulated samples
sim_U = [];   % Store simulated pseudo-observations (0-1 scale)
model_info = [];  % Store model information

switch method_choice
    case 1
        % Method 1: Gaussian/t Copula
        [sim_U, model_info] = gaussian_t_copula_model(Uhat, Nsim);
        fprintf('\n=== Method 1 Results ===\n');
        fprintf('Chosen Copula Type: %s\n', model_info.chosen_type);
        fprintf('Gaussian BIC: %.3f, t Copula BIC: %.3f\n', model_info.BIC_Gauss, model_info.BIC_t);
        
    case 2       
    % Method 2: DTC Sampling
    [sim_U, model_info] = copula_gibbs_model(Uhat, Nsim);

    fprintf('\n==================== Method 2 Results ====================\n');

    % root
    fprintf('Root Variable: %d\n\n', model_info.root);

    % copula edges
    fprintf('Selected Copula Edges and Fitted Copulas:\n');

    np = size(model_info.copula_pairs,1);

    for k = 1:np
        i = model_info.copula_pairs(k,1);
        j = model_info.copula_pairs(k,2);

        fprintf('  Edge %d : (%d , %d)\n', k, i, j);
        fprintf('           Copula = %s\n', model_info.opt_types{k});

        if isfield(model_info,'opt_params')
            params = model_info.opt_params{k};
            fprintf('           Params = ');
            disp(params);
        end
    end

    fprintf('=========================================================\n\n');

end
%%Compare correlation structure of simulated CDF
fprintf('\n--- Simulated CDF Correlation Matrix (before conversion) ---\n');
R_sim_cdf = corrcoef(sim_U);
disp(R_sim_cdf);

%% 7. Convert Simulated Pseudo-Observations back to Original Data Space
sim_data = zeros(size(sim_U));
for i=1:d
    pd = marg_fits{i}.pd;
    fam = marg_fits{i}.family;
    if strcmp(fam, 'GEV')
        sim_data(:,i) = gevinv(sim_U(:,i), pd(1), pd(2), pd(3));
    elseif strcmp(fam, 'GPD')
        sim_data(:,i) = gpdinv(sim_U(:,i), pd(1), pd(2), pd(3));
    elseif isstruct(pd)
        sim_data(:,i) = pd.icdf(sim_U(:,i));
    else
        sim_data(:,i) = icdf(pd, sim_U(:,i));
    end
end


%% 8. Compare Statistics
fprintf('\n--- Mean and Standard Deviation Comparison ---\n');
for j=1:d
    fprintf('Var%d Mean: Original=%.3f Simulated=%.3f | Std Dev: Original=%.3f Simulated=%.3f\n',...
        j,mean(data(:,j)),mean(sim_data(:,j)),std(data(:,j)),std(sim_data(:,j)));
end


%% 9. Compare Correlation Structure
R_orig = corrcoef(data);
R_sim = corrcoef(sim_data);
fprintf('\n--- Correlation Structure Comparison ---\n');
fprintf('Original Data Correlation Matrix:\n');
disp(R_orig);
fprintf('Simulated Data Correlation Matrix:\n');
disp(R_sim);


%% 10. Scatter Plot Comparison
pairs = nchoosek(1:d,2);
figure('Position', [200 200 900 600]);
alpha_val = 0.3;
original_color = [0 0.447 0.741];
simulated_color = [0.85 0.333 0.1];

for k=1:6
    subplot(2,3,k);
    i = pairs(k,1); j = pairs(k,2);
    scatter(data(:,i), data(:,j), 30, original_color, 'filled', ...
        'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', alpha_val);
    hold on;
    scatter(sim_data(:,i), sim_data(:,j), 30, simulated_color, 'filled', ...
        'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', alpha_val);
    xlabel(sprintf('X%d',i), 'FontWeight', 'bold'); 
    ylabel(sprintf('X%d',j), 'FontWeight', 'bold');
    title(sprintf('X%d vs X%d',i,j), 'FontWeight', 'bold');
    grid on;
    box on;
    if k == 1
        legend({'Original Data', 'Simulated Data'}, 'Location', 'best', 'Box', 'off');
    end
end

if method_choice == 1
    method_name = 'Gaussian/t Copula';
else
    method_name = 'DTC';
end
sgtitle(['Scatter Plot Comparison: Original Data vs ', method_name, ' Simulated Data'], 'FontSize', 14, 'FontWeight', 'bold');

% Reset graphics default settings
set(groot, 'DefaultLineLineWidth', 0.5);
set(groot, 'DefaultAxesFontName', 'Helvetica');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Helvetica');


%% Helper functions for Pearson3 and GPD 
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

function [param_gpd, p_gpd] = gpdfit(x, theta)
x_exceed = x(x > theta);
if isempty(x_exceed)
    error('No data exceeds the GPD threshold, please adjust theta.');
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
    x(idx) = theta + (sigma / k) * ( (1 - p(idx)).^(-k) - 1 );
end
end