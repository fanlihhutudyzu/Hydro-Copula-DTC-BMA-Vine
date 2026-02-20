%% scenario_loop_vine_bma_4d_v3.m
% 每个场景运行一次（合成样本 1000x4），BMA 模拟 1000x4 并输出详尽结果与图
% Oct 18 2025
clear; clc; close all;
rng(1); % reproducibility

%% ========== 设置 ==========
scenarios = {'A','B','C'};    % A: t-copula tail, B: mixed arch+gauss blocks, C: mixture-per-observation
n = 1000;                     % 固定合成样本规模 1000 x 4
d = 4;                        % 维度
Nsim = 1000;                  % 模拟样本规模 (1000 x 4)

%% families - 修改为水文常用的5种分布
marg_families = {'Gamma','Lognormal','Pearson3','GEV','Weibull'};  
copula_families = {'Gaussian','t','Clayton','Gumbel','Frank'};

%% 图形样式设置
set(groot, 'DefaultLineLineWidth', 1.2);
set(groot, 'DefaultAxesFontName', 'Times New Roman');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Times New Roman');
set(groot, 'DefaultTextFontSize', 10);
set(groot, 'DefaultLegendFontSize', 9);

%% 散点图样式参数（半透明设置）
original_color = [0 0.447 0.741];    % 蓝色（RGB三元组）
simulated_color = [0.85 0.333 0.1];  % 橙色（RGB三元组）
alpha_value = 0.3;                   % 透明度（0-1）
original_marker_size = 30;
simulated_marker_size = 30;          % 统一大小便于对比

%% ========== 三个场景逐一处理 ==========
for sIdx = 1:length(scenarios)
    scen = scenarios{sIdx};
    fprintf('\n====== Scenario %s ======\n', scen);
    try
        %% 1) 生成 1000x4 合成样本
        data = generate_scenario_data(scen, n); % 返回 n x 4
        
        %% 2) 边缘拟合（BIC 选择） & K-S 检验 & 伪观测 Uhat
        marg_fits = cell(1,d);
        KS_pvals = nan(1,d);
        KS_results = cell(1,d);  % 存储检验结果
        Uhat = zeros(n,d);

        for i = 1:d
            x = data(:,i);
            bestBIC = Inf;
            bestFit = [];
            bestFam = '';
            
            % 拟合所有候选分布，选择最优BIC模型
            for f = marg_families
                fam = f{1};
                try
                    switch fam
                        case {'Gamma', 'Lognormal', 'Weibull'}
                            pd = fitdist(x, fam);  % 内置分布
                            p = 2;
                        case 'Pearson3'
                            pd = pearson3fit(x);  % 自定义分布
                            p = 3;
                        case 'GEV'  % GEV分布处理
                            [shape, loc, scale] = gevfit(x);
                            pd = struct('shape', shape, 'loc', loc, 'scale', scale);
                            p = 3;
                    end
                    
                    % 计算BIC
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
                    
                    % 更新最优模型
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
            
            % 计算伪观测Uhat和K-S检验
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
                        error('CDF矩阵构造失败');
                    end
                    
                    [h, pval] = kstest(x, 'CDF', cdf_matrix, 'Alpha', 0.05);
                    Uhat(:,i) = cdf_func(x);
                    KS_pvals(i) = pval;
                    KS_results{i} = sprintf('%s (p=%.4f)', (h==0)*'Passed' + (h==1)*'Failed', pval);
                catch ME
                    warning('变量 %d 的K-S检验失败: %s', i, ME.message);
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
        
        %% 3) 生成 unique C-vine 与 D-vine orders
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
        
        %% 4) 拟合所有 unique C-vine 和 D-vine
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
        
        % 合并模型并收集可用模型信息
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
            error('没有成功拟合任何 Vine 模型');
        end
        
        % 提取 BIC 列表与 orders/types
        allBICs = [model_info.BIC];
        allOrders = cellfun(@(o) sprintf('%d-', o{:}), arrayfun(@(m) {m.order}, model_info, 'UniformOutput', false), 'UniformOutput', false);
        % 构造字符串
        AllOrders_strs = cell(1,Num_Valid_Models);
        AllTypes = cell(1,Num_Valid_Models);
        for ii=1:Num_Valid_Models
            ordv = model_info(ii).order;
            AllOrders_strs{ii} = sprintf('%d-', ordv);
            AllOrders_strs{ii}(end) = []; % remove trailing '-'
            AllTypes{ii} = model_info(ii).type;
        end
        
        % 计算 BMA 权重（基于 BIC）- 补全allW的定义
        logL_rel = -0.5 * allBICs;
        logL_rel = logL_rel - max(logL_rel); % 稳定化
        w_num = exp(logL_rel);
        allW = w_num / sum(w_num);  % 补全此行，定义allW
        
        % 找出最优 C 与 D（如果存在）
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
        
        % 结果字符串拼接
        AllBICs_str = strjoin(arrayfun(@(v) sprintf('%.2f',v), allBICs, 'UniformOutput', false), ';');
        AllW_str   = strjoin(arrayfun(@(v) sprintf('%.4f',v), allW, 'UniformOutput', false), ';');
        AllOrders_str = strjoin(AllOrders_strs, ';');
        AllTypes_str  = strjoin(AllTypes, ';');
        KS_pvals_str  = strjoin(arrayfun(@(v) sprintf('%.4f',v), Marginal_KS_Pvalues, 'UniformOutput', false), ';');
        KS_results_str = strjoin(KS_results, ';');  % K-S检验结果字符串
        
        %% 5) 基于 BMA 权重做仿真
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
        
        % 逆变换到原始空间
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
        
        %% 6) 显示结果
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
        
        % 相关系数矩阵
        R_orig = corr(data, 'Type', 'Pearson');
        R_sim  = corr(simBMA_data, 'Type', 'Pearson');
        disp('  Pearson correlation matrix (original):'); disp(R_orig);
        disp('  Pearson correlation matrix (simulated):'); disp(R_sim);
        
        Tau_orig = corr(Uhat, 'Type', 'Kendall');
        Tau_sim  = corr(simU, 'Type', 'Kendall');
        disp('  Kendall correlation matrix (original):'); disp(Tau_orig);
        disp('  Kendall correlation matrix (simulated):'); disp(Tau_sim);
        
        %% 8) 绘图
        fig = figure('Name', ['Scenario ', scen], 'Units','normalized','Position',[0.1 0.1 0.9 0.7]);
        fig.Color = [1 1 1]; % 白色背景
        pairs = nchoosek(1:d,2);
        
        % QQ图 (前4个)
        for i=1:4
            subplot(2,5,i);
            x = data(:,i);
            pd = marg_fits{i}.pd;
            fam = marg_fits{i}.family;
            nobs = length(x);
            p = (1:nobs)'/(nobs+1);
            
            % 计算理论分位数
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
            
            % 绘制QQ图
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
        
        % 散点对比 (后6个)
        for k=1:6
            subplot(2,5,4+k);
            i = pairs(k,1); j = pairs(k,2);
            
            % 原始数据散点（半透明蓝色）
            scatter(data(:,i), data(:,j), original_marker_size, ...
                original_color, 'filled', ...
                'MarkerEdgeColor', 'none', ...
                'MarkerFaceAlpha', alpha_value);
            
            hold on;
            
            % 模拟数据散点（半透明橙色）
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
        
        % 添加总标题
        sgtitle(sprintf('Scenario %s: Marginal QQ Plots and Scatter Comparisons', scen), ...
            'FontSize', 14, 'FontWeight', 'bold');
        
                
    catch ME
        warning('Scenario %s failed: %s', scen, ME.message);
    end
end

% 重置图形默认设置
set(groot, 'DefaultLineLineWidth', 0.5);
set(groot, 'DefaultAxesFontName', 'Helvetica');
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultTextFontName', 'Helvetica');
set(groot, 'DefaultTextFontSize', 10);
set(groot, 'DefaultLegendFontSize', 9);