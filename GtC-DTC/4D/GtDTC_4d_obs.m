%% GtG_4D_obs_AICc.m
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
    bestAICc = Inf;
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
            
            % Calculate log-likelihood and AICc
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
            AIC = -2*logL + 2*p;
            AICc = AIC + 2*p*(p+1)/(n-p-1);
            
            % Update best model
            if AICc < bestAICc
                bestAICc = AICc; 
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

    % 新增：计算每个变量的 KS 检验 p值
    x = data(:,i);
    pd = bestFit;
    fam = bestFam;
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

%% 4. Marginal Distribution Diagnosis: QQ Plot —— 已按要求修改
set(groot, 'DefaultLineLineWidth', 1.5);
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultAxesFontSize', 14);  % 全局字体14
set(groot, 'DefaultTextFontName', 'Times New Roman');

% 真实变量名称
var_names = {'Yangzhou','Yizheng','Jiangdu','Gaoyou'};

figure('Name', 'Marginal Distribution QQ Plots', 'Position', [100 100 900 800]);
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
    plot(q_theo, q_emp, 'o', 'MarkerSize', 5, 'MarkerEdgeColor', [0.2 0.2 0.8], 'MarkerFaceColor', [0.7 0.7 1]);
    hold on;
    min_val = min([q_theo; q_emp]);
    max_val = max([q_theo; q_emp]);
    line([min_val, max_val], [min_val, max_val], 'Color', [0.8 0 0], 'LineStyle', '--','LineWidth',1.5);
    
    xlabel(sprintf('Theoretical Quantiles (%s)', fam), 'FontWeight', 'bold','FontSize',14);
    ylabel('Sample Quantiles', 'FontWeight', 'bold','FontSize',14);
    title(var_names{i}, 'FontWeight', 'bold','FontSize',14);  % 真实变量名
    grid on;
    box on;

    % 新增：标注 K-S 检验结果
    ks_p = marg_fits{i}.ks_p;
    text_str = sprintf('K-S test: p-value=%.3f', ks_p);
    text(0.05, 0.95, text_str, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'FontSize', 12, ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');
end
sgtitle('Anual Precipitation (mm)', 'FontSize', 16, 'FontWeight', 'bold');  % 总标题16号


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
        fprintf('Gaussian AICc: %.3f, t Copula AICc: %.3f\n', model_info.AICc_Gauss, model_info.AICc_t);
        
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


%% 10. Scatter Plot Comparison —— 已按要求修改字体
pairs = nchoosek(1:d,2);
figure('Position', [200 200 1000 700]);
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
    xlabel(var_names{i}, 'FontWeight', 'bold','FontSize',14);  % 真实变量名
    ylabel(var_names{j}, 'FontWeight', 'bold','FontSize',14);
    title([var_names{i} ' vs ' var_names{j}], 'FontWeight', 'bold','FontSize',14);
    grid on;
    box on;
    if k == 1
        legend({'Original Data', 'Simulated Data'}, 'Location', 'best', 'Box', 'off','FontSize',12);
    end
end

if method_choice == 1
    method_name = 'Gaussian/t Copula';
else
    method_name = 'DTC';
end
sgtitle(['Scatter Plot Comparison: Original Data vs ', method_name, ' Simulated Data'], 'FontSize', 16, 'FontWeight', 'bold');

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

%% Gaussian/t Copula Model (AICc版本)
function [sim_data, model_info] = gaussian_t_copula_model(U, m)
[n,d] = size(U);

% Fit Gaussian copula
R_gauss = copulafit('Gaussian', U);
ll_gauss = sum(log(copulapdf('Gaussian', U, R_gauss)));
k_gauss = d*(d-1)/2;
AICc_Gauss = -2*ll_gauss + 2*k_gauss + 2*k_gauss*(k_gauss+1)/(n-k_gauss-1);

% Fit t copula
try
    [R_t, nu_t] = copulafit('t', U);
catch ME
    warning(ME.identifier,'t copula fitting failed: %s. Using default nu=4.', ME.message);
    nu_t = 4;
    R_t = copulafit('Gaussian', U);
end

ll_t = sum(log(copulapdf('t', U, R_t, nu_t)));
k_t = d*(d-1)/2 + 1;
AICc_t = -2*ll_t + 2*k_t + 2*k_t*(k_t+1)/(n-k_t-1);

% Select best model
if AICc_t < AICc_Gauss
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
    'AICc_Gauss',AICc_Gauss,'AICc_t',AICc_t,'chosen_type',chosen,'chosen_params',params);
end

%% DTC Gibbs Sampling Model (AICc版本)
function [gibbs_sim_data, model_info] = copula_gibbs_model(U, m)
[n,d] = size(U);
corr_mat = corr(U,'type','Kendall');
abs_corr = abs(corr_mat); abs_corr(1:d+1:end)=0;

% Build tree
pairs_list = [];
for i=1:d, for j=i+1:d, pairs_list=[pairs_list; abs_corr(i,j),i,j]; end, end
pairs_list = sortrows(pairs_list,1,'descend');
copula_pairs = []; used_nodes = [];
for k=1:size(pairs_list,1)
    a = pairs_list(k,2); b = pairs_list(k,3);
    if isempty(used_nodes)
        copula_pairs = [copula_pairs; a,b]; used_nodes = [a,b];
    else
        if sum(ismember(a,used_nodes)) + sum(ismember(b,used_nodes)) ==1
            copula_pairs = [copula_pairs; a,b]; used_nodes = unique([used_nodes,a,b]);
        end
    end
    if length(used_nodes)==d, break; end
end
root = copula_pairs(1,1);

% Fit pairwise copulas
candidate_types = {'Gaussian','t','Frank','Gumbel','Clayton'};
num_pairs = size(copula_pairs,1);
opt_types = cell(1,num_pairs); opt_params = cell(1,num_pairs);
for k=1:num_pairs
    upair = U(:,copula_pairs(k,:));
    [best_type, best_params] = fit_pairwise_copula(upair, candidate_types);
    opt_types{k}=best_type; opt_params{k}=best_params;
end

% Gibbs sampling
Usim = zeros(m,d);
for sim=1:m
    u = zeros(1,d); done = false; cnt=0;
    while ~done && cnt<100
        try
            u = zeros(1,d); u(root)=rand;
            for k=1:size(copula_pairs,1)
                a = copula_pairs(k,1); b = copula_pairs(k,2);
                if u(a)~=0 && u(b)==0
                    w=rand; typ=opt_types{k}; params=opt_params{k};
                    v = hinv_conditional(typ,params,u(a),w,1);
                    u(b)=min(max(v,1e-12),1-1e-12);
                elseif u(b)~=0 && u(a)==0
                    w=rand; typ=opt_types{k}; params=opt_params{k};
                    v = hinv_conditional(typ,params,u(b),w,2);
                    u(a)=min(max(v,1e-12),1-1e-12);
                end
            end
            done = true;
        catch
            cnt=cnt+1;
        end
    end
    Usim(sim,:)=u;
end

gibbs_sim_data = Usim;
model_info = struct('root',root,'corr_mat',corr_mat,'copula_pairs',copula_pairs,...
    'opt_types',{opt_types},'opt_params',{opt_params});
end

%% Pairwise Copula Fitting (AICc版本)
function [best_type, best_params] = fit_pairwise_copula(u_pair, candidate_types)
n = size(u_pair,1);
best_AICc = Inf; best_type = ''; best_params = [];
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
        AIC = -2*ll + 2*k;
        AICc = AIC + 2*k*(k+1)/(n-k-1);
        if AICc < best_AICc
            best_AICc = AICc;
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

%% Conditional Inverse Functions
function v = hinv_conditional(type, params, u_cond, w, idx_in_pair_cond)
if idx_in_pair_cond == 1
    if strcmpi(type,'Gaussian')
        rho = params.R(1,2); v = hinv_gaussian(w, u_cond, rho);
    elseif strcmpi(type,'t')
        rho = params.R(1,2); nu = params.nu; v = hinv_t(w, u_cond, rho, nu);
    else
        f = @(v) numeric_partial_h(type, params, u_cond, v, 1) - w;
        v = hinv_numeric(f);
    end
else
    if strcmpi(type,'Gaussian')
        rho = params.R(1,2); v = hinv_gaussian(w, u_cond, rho);
    elseif strcmpi(type,'t')
        rho = params.R(1,2); nu = params.nu; v = hinv_t(w, u_cond, rho, nu);
    else
        f = @(v) numeric_partial_h(type, params, v, u_cond, 2) - w;
        v = hinv_numeric(f);
    end
end
end

function u_j = hinv_numeric(f)
lb = 1e-8; ub = 1-1e-8; x0=0.5;
try
    if f(lb)*f(ub)>0
        grid=linspace(lb,ub,9); vals=arrayfun(f,grid);
        [~,idx]=min(abs(vals)); x0=grid(idx);
    end
    u_j=fzero(f,x0);
catch
    u_j=(lb+ub)/2;
end
u_j=min(max(u_j,1e-12),1-1e-12);
end

function h = numeric_partial_h(type, params, u, v, w_r_t)
epsu=1e-6;
if w_r_t==1
    u1=min(max(u+epsu,0),1); u0=min(max(u-epsu,0),1);
    uu1=[u1,v]; uu0=[u0,v];
else
    v1=min(max(v+epsu,0),1); v0=min(max(v-epsu,0),1);
    uu1=[u,v1]; uu0=[u,v0];
end
denom=2*epsu;
try
    if strcmpi(type,'Gaussian')
        C1=copulacdf('Gaussian',uu1,params.R); C0=copulacdf('Gaussian',uu0,params.R);
    elseif strcmpi(type,'t')
        C1=copulacdf('t',uu1,params.R,params.nu); C0=copulacdf('t',uu0,params.R,params.nu);
    else
        C1=copulacdf(type,uu1,params.theta); C0=copulacdf(type,uu0,params.theta);
    end
    h=(C1-C0)/denom;
catch
    h=0;
end
end

function v = hinv_gaussian(w, u, rho)
z_u=norminv(u); z_w=norminv(w);
z_v=rho*z_u+sqrt(max(0,1-rho^2))*z_w;
v=normcdf(z_v);
end

function v = hinv_t(w, u, rho, nu)
z_u=tinv(u,nu);
mean_c=rho*z_u;
scale_c=sqrt((nu+z_u.^2)/(nu+1)*(1-rho^2));
t_w=tinv(w,nu+1);
t_v=mean_c+scale_c*t_w;
v=tcdf(t_v,nu);
end