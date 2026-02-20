%% vine_bma_4d_obs.m
% 4D 实测值 + 边缘拟合(含GPD) + C/D vine  + BMA 加权 + 统计对比 + 相关性对比

clear; clc; close all; rng(1);

%% 1. 输入数据
% load('Storm_NL_4D.mat')
% data = storm_NL_4D;
load preci_yangzhou.mat;
data = preci_yangzhou;
n = length(data);
d = 4;

%% 2. 边缘拟合与优选
families = {'Gamma','Lognormal','Pearson3','GEV','Weibull','GPD'};
for i=1:d
    x = data(:,i);
    bestBIC = Inf;
    bestFit_theta_gpd = NaN;
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
    [~, idx] = sort(x);
    ranks = zeros(n,1);
    ranks(idx) = (1:n)';
    Uhat(:,i) = ranks / (n + 1);
end
Uhat = max(realmin, min(1-realmin, Uhat));

% 计算原始经验分布相关矩阵
R_empirical_Pearson = corrcoef(Uhat);
R_empirical_Kendall = corr(Uhat, 'Type', 'Kendall');

fprintf('\n--- Original Empirical Distribution Correlation Matrix ---\n');
fprintf('Pearson:\n'); disp(R_empirical_Pearson);
fprintf('Kendall:\n'); disp(R_empirical_Kendall);

%% 4. 边缘分布诊断：QQ图和拟合优度
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
    line([min([q_theo;q_emp]), max([q_theo;q_emp])],[min([q_theo;q_emp]), max([q_theo;q_emp])],'Color',[0.8 0 0],'LineStyle','--');
    xlabel(sprintf('Theoretical Quantiles (%s)', fam), 'FontWeight', 'bold');
    ylabel('Sample Quantiles', 'FontWeight', 'bold');
    title(sprintf('Variable %d', i), 'FontWeight', 'bold');
    grid on; box on;
end
sgtitle('Marginal Distribution QQ Plots', 'FontSize', 14, 'FontWeight', 'bold');

%% 5. 定义 Copula 候选族
copula_families = {'Gaussian','t','Clayton','Gumbel','Frank'};

%% 6. C-vine 与 D-vine 结构优选 
orders_all = perms(1:d);
unique_C_orders = {}; 
unique_D_orders = {}; 

% 这里保留逻辑但不打印结构数量和匹配信息
for i=1:size(orders_all,1)
    ord = orders_all(i,:);
    ord_swap_tail = ord; if d>2, ord_swap_tail([d-1,d])=ord_swap_tail([d,d-1]); end
    is_C_unique = true;
    for k=1:length(unique_C_orders), if isequal(ord,unique_C_orders{k})||isequal(ord_swap_tail,unique_C_orders{k}), is_C_unique=false; break; end; end
    if is_C_unique, unique_C_orders{end+1} = ord; end
end
for i=1:size(orders_all,1)
    ord = orders_all(i,:);
    ord_rev = flip(ord);
    is_D_unique = true;
    for k=1:length(unique_D_orders), if isequal(ord,unique_D_orders{k})||isequal(ord_rev,unique_D_orders{k}), is_D_unique=false; break; end; end
    if is_D_unique, unique_D_orders{end+1} = ord; end
end

% 拟合C/D vine模型
unique_C_models = cell(1,length(unique_C_orders));
unique_D_models = cell(1,length(unique_D_orders));
bestC.BIC = Inf; bestD.BIC = Inf;

for i=1:length(unique_C_orders)
    ord = unique_C_orders{i};
    try modelC = fit_four_dim_cvine(Uhat, ord, copula_families); modelC.type='C'; unique_C_models{i}=modelC; if modelC.BIC<bestC.BIC, bestC=modelC; end
    catch, continue; end
end
for i=1:length(unique_D_orders)
    ord = unique_D_orders{i};
    try modelD = fit_four_dim_dvine(Uhat, ord, copula_families); modelD.type='D'; unique_D_models{i}=modelD; if modelD.BIC<bestD.BIC, bestD=modelD; end
    catch, continue; end
end
all_models = [unique_C_models, unique_D_models];

%% 7. BMA 权重
num_models = length(all_models);
if num_models==0, error('No unique C-vine or D-vine models were successfully fitted for BMA.'); end

BICs=zeros(1,num_models);
for k=1:num_models, BICs(k)=all_models{k}.BIC; end
logL_rel=-0.5*BICs; max_logL_rel=max(logL_rel); w_numerator=exp(logL_rel-max_logL_rel); w=w_numerator/sum(w_numerator);

%% 显示所有模型权重、顺序及类型（简化版）
fprintf('\n=== All BMA Model Details (Total %d Models) ===\n', num_models);
for k = 1:num_models
    mdl = all_models{k};
    fprintf('Model %2d: type=%s, order=%s, BIC=%.3f, Weight=%.5f\n', ...
        k, mdl.type, mat2str(mdl.order), BICs(k), w(k));
end

%% 8. 仿真
Nsim=1000; simBMA=[];
for k=1:num_models
    Nk=round(Nsim*w(k));
    if Nk>0
        model=all_models{k};
        if strcmp(model.type,'C')
            simk=simulate_four_dim_cvine_fixed(Nk, model);
        else
            simk=simulate_four_dim_dvine_fixed(Nk, model);
        end
        simBMA=[simBMA; simk];
    end
end
Nsim_actual = size(simBMA,1);

%% 模拟CDF相关矩阵
R_sim_cdf_Pearson = corrcoef(simBMA);
R_sim_cdf_Kendall = corr(simBMA,'Type','Kendall');
fprintf('\n--- Simulated CDF Correlation Matrix (before conversion) ---\n');
fprintf('Pearson:\n'); disp(R_sim_cdf_Pearson);
fprintf('Kendall:\n'); disp(R_sim_cdf_Kendall);

%% 转换回原始数据空间
simBMA_data = zeros(size(simBMA));
for i=1:d
    pd = marg_fits{i}.pd; fam = marg_fits{i}.family;
    if strcmp(fam,'GEV')
        simBMA_data(:,i)=gevinv(simBMA(:,i),pd(1),pd(2),pd(3));
    elseif strcmp(fam,'GPD')
        simBMA_data(:,i)=gpdinv(simBMA(:,i),pd(1),pd(2),pd(3));
    elseif isstruct(pd)
        simBMA_data(:,i)=pd.icdf(simBMA(:,i));
    else
        simBMA_data(:,i)=icdf(pd,simBMA(:,i));
    end
end
simBMA = simBMA_data;

%% 9. 比较统计量
fprintf('\n--- Comparing Means and Standard Deviations ---\n');
for j=1:d
    fprintf('Var%d mean: orig=%.3f sim=%.3f | std: orig=%.3f sim=%.3f\n',...
        j,mean(data(:,j)),mean(simBMA(:,j)),std(data(:,j)),std(simBMA(:,j)));
end

%% 10. 比较相关性结构
R_orig_Pearson=corrcoef(data); R_orig_Kendall=corr(data,'Type','Kendall');
R_sim_Pearson=corrcoef(simBMA); R_sim_Kendall=corr(simBMA,'Type','Kendall');

fprintf('\n--- Comparing Correlation Structures ---\n');
fprintf('Original data sample correlation matrix:\nPearson:\n'); disp(R_orig_Pearson);
fprintf('Kendall:\n'); disp(R_orig_Kendall);
fprintf('BMA simulated data sample correlation matrix:\nPearson:\n'); disp(R_sim_Pearson);
fprintf('Kendall:\n'); disp(R_sim_Kendall);

%% 11. 散点对比 
pairs = nchoosek(1:d,2);
figure('Position', [200 200 900 600]);
% 定义半透明参数和颜色
alpha_val = 0.3;                  % 透明度（0-1之间）
original_color = [0 0.447 0.741]; % 蓝色（原始数据）
simulated_color = [0.85 0.333 0.1];% 橙色（模拟数据）

for k=1:6
    subplot(2,3,k);
    i = pairs(k,1); j = pairs(k,2);
    % 原始数据散点（半透明蓝色）
    scatter(data(:,i), data(:,j), 30, original_color, 'filled', ...
        'MarkerEdgeColor', 'none', ...  % 去除边缘线
        'MarkerFaceAlpha', alpha_val);  % 设置填充透明度
    
    hold on;
    % 模拟数据散点（半透明橙色）
    scatter(simBMA(:,i), simBMA(:,j), 30, simulated_color, 'filled', ...
        'MarkerEdgeColor', 'none', ...  % 去除边缘线
        'MarkerFaceAlpha', alpha_val);  % 设置填充透明度
    
    xlabel(sprintf('X%d',i), 'FontWeight', 'bold'); 
    ylabel(sprintf('X%d',j), 'FontWeight', 'bold');
    title(sprintf('X%d vs X%d',i,j), 'FontWeight', 'bold');
    grid on;
    box on;
    % 添加图例区分数据类型
    if k == 1  % 只在第一个子图添加图例
        legend({'Original', 'Simulated'}, 'Location', 'best', 'Box', 'off');
    end
end
sgtitle('Scatter Plots: Original vs BMA Simulated', 'FontSize', 14, 'FontWeight', 'bold');

% 重置图形默认设置
set(groot, 'DefaultLineLineWidth', 0.5);       % 线条默认宽度（原始默认值）
set(groot, 'DefaultAxesFontName', 'Helvetica'); % 坐标轴默认字体（原始默认值，视系统而定）
set(groot, 'DefaultAxesFontSize', 10);          % 坐标轴默认字号（原始默认值）
set(groot, 'DefaultTextFontName', 'Helvetica'); % 文本默认字体（原始默认值，视系统而定）


% GPD分布拟合函数（k, sigma, theta）
function [param_gpd, p_gpd] = gpdfit(x, theta)
    % 提取超过阈值的数据（GPD仅拟合超额量）
    x_exceed = x(x > theta);
    
    if isempty(x_exceed)
        error('No data exceeds the GPD threshold. Adjust theta.');
    end
    
    % 拟合GPD的形状参数k和尺度参数sigma（使用广义帕累托分布拟合函数）
    param_hat = gpfit(x_exceed - theta);  % 拟合超额量 x - theta
    
    % param_hat = [k, sigma]，组合为[k, sigma, theta]
    k_hat = param_hat(1);
    sigma_hat = param_hat(2);
    param_gpd = [k_hat, sigma_hat, theta]; 
    
    p_gpd = 2;  % GPD有2个待估参数（k和sigma）
end

% GPD分布PDF函数
function y = gpdpdf(x, k, sigma, theta)
    x_exceed = x - theta;  % 计算超额量
    y = zeros(size(x));    % 初始化PDF结果
    
    % 仅对超额量为正的部分计算PDF
    idx = x_exceed > 0;
    
    if k == 0  % 特殊情况：k=0时退化为指数分布
        y(idx) = (1/sigma) * exp(-x_exceed(idx) / sigma);
    else  % 一般情况
        factor = 1 + k * x_exceed(idx) / sigma;
        valid_idx = idx & (factor > 0);  % 确保因子为正（分布有效性约束）
        y(valid_idx) = (1/sigma) * (factor(valid_idx)).^(-1/k - 1);
    end
end

% GPD分布CDF函数
function F = gpdcdf(x, k, sigma, theta)
    x_exceed = x - theta;  % 计算超额量
    F = zeros(size(x));    % 初始化CDF结果
    
    % 仅对超额量为正的部分计算CDF
    idx = x_exceed > 0;
    
    if k == 0  % 指数分布情况
        F(idx) = 1 - exp(-x_exceed(idx) / sigma);
    else  % 一般情况
        factor = 1 + k * x_exceed(idx) / sigma;
        valid_idx = idx & (factor > 0);  % 分布有效性约束
        F(valid_idx) = 1 - (factor(valid_idx)).^(-1/k);
    end
    F = max(0, min(1, F));  % 确保CDF在[0,1]范围内
end

% GPD分布分位数函数（ICDF）
function x = gpdinv(p, k, sigma, theta)
    x = zeros(size(p));  % 初始化分位数结果
    
    % 仅对有效概率范围计算分位数
    idx = p > 0 & p < 1;
    
    if k == 0  % 指数分布情况
        x(idx) = theta - sigma * log(1 - p(idx));
    else  % 一般情况
        x(idx) = theta + (sigma / k) * ( (1 - p(idx)).^(-k) - 1 );
    end
end