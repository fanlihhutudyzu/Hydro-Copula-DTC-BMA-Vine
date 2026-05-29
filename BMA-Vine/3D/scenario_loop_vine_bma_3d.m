clear; clc; close all; rng(1);

% ---------------- top-level settings ----------------
scenarios = {'A','B','C'};
n = 1000;  % sample size
families = {'Gaussian','t','Clayton','Gumbel','Frank'}; % candidate copulas
unique_vine_orders = {[1,2,3],[2,1,3],[3,1,2]};
Nsim = 1000;  % BMA simulation total

% ---------------- main loop over scenarios ----------------
for s = 1:length(scenarios)
    scenario = scenarios{s};
    fprintf('\n=============================================\n');
    fprintf('           Analyzing Scenario %s            \n', scenario);
    fprintf('=============================================\n');

    % 1) generate data
    data = generate_scenario_3Ddata(scenario, n);
    d = size(data,2);

    % 2) marginal fitting (Gamma, Lognormal, Weibull, Pearson3, GEV)
    marg_fits = cell(1,d);
    for i = 1:d
        x = data(:,i);
        families_m = {'Gamma','Lognormal','Weibull','Pearson3','GEV'};
        bestAICc = Inf; bestFit=[]; bestFam='';
        for fm = families_m
            fam = fm{1};
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
                        parmhat = gevfit(x); p = 3;
                        pd = parmhat;
                end
                if strcmp(fam,'GEV')
                    pdfv = gevpdf(x, pd(1), pd(2), pd(3));
                elseif isstruct(pd)
                    pdfv = pd.pdf(x);
                else
                    pdfv = pdf(pd, x);
                end
                pdfv(pdfv<=0) = realmin;
                logL = sum(log(pdfv));
                AIC = -2*logL + 2*p;
                AICc = AIC + 2*p*(p+1)/(n-p-1);
                if AICc < bestAICc
                    bestAICc = AICc; bestFit = pd; bestFam = fam;
                end
            catch
                continue;
            end
        end
        marg_fits{i}.pd = bestFit;
        marg_fits{i}.family = bestFam;
        fprintf('Var%d: best marginal = %s\n', i, bestFam);
    end

    % 3) pseudo-observations (ranks/(n+1))
    Uhat = zeros(n,d);
    for i = 1:d
        [~, idx] = sort(data(:,i));
        ranks = zeros(n,1);
        ranks(idx) = (1:n)';
        Uhat(:,i) = ranks/(n+1);
    end
    Uhat = max(realmin, min(1-realmin, Uhat));
    R_empirical = corrcoef(Uhat);
    fprintf('\n--- Original Empirical Distribution Correlation Matrix ---\n');
    disp(R_empirical);

    % 4) fit vine for each unique order (use corrected fit)
    vine_models = cell(1,length(unique_vine_orders));
    bestVine.AICc = Inf;
    for iord = 1:length(unique_vine_orders)
        ord = unique_vine_orders{iord};
        try
            model = fit_three_dim_vine_correct(Uhat, ord, families);
            vine_models{iord} = model;
            if model.AICc < bestVine.AICc
                bestVine = model;
            end
        catch ME
            fprintf('Vine fit failed for order %s: %s\n', mat2str(ord), ME.message);
            vine_models{iord} = [];
        end
    end

    % display details
    fprintf('\nUsing %d unique Vine structures for 3D case.\n', length(unique_vine_orders));
    fprintf('\n=== Detailed Copula Information for Each Vine Structure ===\n');
    valid_models = {};
    for k = 1:length(vine_models)
        model = vine_models{k};
        if isempty(model) || ~isfield(model,'order'), continue; end
        valid_models{end+1} = model;
        ord = model.order;
        fprintf('\nVine Structure %d: Order = %s, AICc = %.3f\n', k, mat2str(ord), model.AICc);
        t1f = model.Trees{1}.fams;
        t2f = model.Trees{2}.fams;
        fprintf('  Tree 1 Copulas:\n');
        fprintf('    - Variable pair (%d, %d): %s\n', ord(1), ord(2), t1f{1});
        fprintf('    - Variable pair (%d, %d): %s\n', ord(1), ord(3), t1f{2});
        fprintf('  Tree 2 Copulas:\n');
        fprintf('    - Conditional pair (%d, %d)|%d: %s\n', ord(2), ord(3), ord(1), t2f{1});
    end

    % best vine
    fprintf('\n=== Best Vine structure (from %d unique vines) ===\n', length(valid_models));
    if isfield(bestVine,'order')
        fprintf('Order: %s, AICc: %.3f\n', mat2str(bestVine.order), bestVine.AICc);
        fprintf('Best Vine Copula Details:\n');
        fprintf('  Tree 1 Copulas:\n');
        fprintf('    - Variable pair (%d, %d): %s\n', bestVine.order(1), bestVine.order(2), bestVine.Trees{1}.fams{1});
        fprintf('    - Variable pair (%d, %d): %s\n', bestVine.order(1), bestVine.order(3), bestVine.Trees{1}.fams{2});
        fprintf('  Tree 2 Copulas:\n');
        fprintf('    - Conditional pair (%d, %d)|%d: %s\n', bestVine.order(2), bestVine.order(3), bestVine.order(1), bestVine.Trees{2}.fams{1});
    else
        fprintf('No Vine model was successfully fitted.\n');
        continue;
    end

    % 5) BMA weights based on AICc
    valid_vine_models = vine_models(~cellfun('isempty',vine_models));
    num_models = length(valid_vine_models);
    if num_models == 0
        fprintf('No models for BMA\n'); continue;
    end
    AICcs = zeros(1,num_models);
    for k = 1:num_models, AICcs(k)=valid_vine_models{k}.AICc; end
    logL_rel = -0.5 * AICcs;
    max_logL_rel = max(logL_rel);
    w_numer = exp(logL_rel - max_logL_rel);
    w = w_numer / sum(w_numer);

    fprintf('\n=== BMA Weights for Each Vine Structure ===\n');
    for k = 1:num_models
        ord = valid_vine_models{k}.order;
        fprintf('Vine Structure (Order %s): AICc = %.3f, Weight = %.5f\n', mat2str(ord), AICcs(k), w(k));
    end
    fprintf('Total AICc Range: [%.3f, %.3f]\n', min(AICcs), max(AICcs));
    fprintf('Sum of weights: %.5f\n', sum(w));

    % 6) simulation using BMA
    simBMA = [];
    for k = 1:num_models
        Nk = round(Nsim * w(k));
        if Nk > 0
            model = valid_vine_models{k};
            simk = simulate_three_dim_vine_correct(Nk, model);
            simBMA = [simBMA; simk];
        end
    end
    Nsim_actual = size(simBMA,1);
    fprintf('Note: Total simulated samples adjusted to Nsim=%d due to rounding.\n', Nsim_actual);

    % transform back to original scale using fitted marginals
    simBMA_data = zeros(size(simBMA));
    for i = 1:d
        pd = marg_fits{i}.pd; fam = marg_fits{i}.family;
        if strcmp(fam,'GEV')
            simBMA_data(:,i) = gevinv(simBMA(:,i), pd(1), pd(2), pd(3));
        elseif isstruct(pd)
            simBMA_data(:,i) = pd.icdf(simBMA(:,i));
        else
            simBMA_data(:,i) = icdf(pd, simBMA(:,i));
        end
    end

    % 7) compare means/std and correlations
    fprintf('\n--- Comparing Means and Standard Deviations ---\n');
    for j = 1:d
        fprintf('Var%d mean: orig=%.3f sim=%.3f | std: orig=%.3f sim=%.3f\n',...
            j, mean(data(:,j)), mean(simBMA_data(:,j)), std(data(:,j)), std(simBMA_data(:,j)));
    end

    R_orig = corrcoef(data);
    R_sim = corrcoef(simBMA_data);
    fprintf('\n--- Comparing Correlation Structures ---\n');
    fprintf('Original data sample correlation matrix:\n'); disp(R_orig);
    fprintf('BMA simulated data sample correlation matrix:\n'); disp(R_sim);

    % optional scatter plots
    pairs = nchoosek(1:d,2);
    figure('Name',sprintf('Scenario %s: Scatter Plots',scenario),'Position',[200 200 1200 400]);
    alpha_val = 0.3;
    original_color = [0 0.447 0.741];
    simulated_color = [0.85 0.333 0.1];
    for k = 1:size(pairs,1)
        subplot(1,3,k);
        i = pairs(k,1); j = pairs(k,2);
        scatter(data(:,i), data(:,j), 30, original_color, 'filled', 'MarkerFaceAlpha', alpha_val); hold on;
        scatter(simBMA_data(:,i), simBMA_data(:,j), 30, simulated_color, 'filled', 'MarkerFaceAlpha', alpha_val);
        xlabel(sprintf('X%d',i)); ylabel(sprintf('X%d',j)); title(sprintf('X%d vs X%d',i,j));
        grid on; box on;
        if k==1, legend({'Original','Simulated'},'Location','best','Box','off'); end
    end
    sgtitle(sprintf('Scenario %s: Original vs BMA Simulated',scenario));
end

% ---------------- end of main script ----------------
% ---------- helper functions (placed below) ------------

%%%--------------- Pearson3 fit ----------------
function pd = pearson3fit(x)
    m = mean(x); s = std(x); g1 = skewness(x);
    if abs(g1) < 1e-6, g1 = 1e-6; end
    k = 4/(g1^2); theta = s/sqrt(k); a = m - k*theta;
    pd.k = k; pd.theta = theta; pd.a = a;
    pd.pdf = @(xx) pearson3_pdf(xx,k,theta,a);
    pd.cdf = @(xx) pearson3_cdf(xx,k,theta,a);
    pd.icdf = @(u) a + gaminv(u,k,theta);
end
function y = pearson3_pdf(x,k,theta,a)
    z = x - a; y = zeros(size(z)); idx = z>0;
    y(idx) = gampdf(z(idx), k, theta);
end
function F = pearson3_cdf(x,k,theta,a)
    z = x - a; F = zeros(size(z)); idx = z>0;
    F(idx) = gamcdf(z(idx), k, theta);
end

%%%--------------- corrected vine functions ----------------
function model = fit_three_dim_vine_correct(U, order, fams)
    n = size(U,1);
    Uord = U(:, order);
    [f12, p12, logL12, k12] = fit_pair_correct(Uord(:,1), Uord(:,2), fams);
    [f13, p13, logL13, k13] = fit_pair_correct(Uord(:,1), Uord(:,3), fams);
    total_logL = logL12 + logL13;
    total_k = k12 + k13;
    U2g1 = cond_cdf(f12, p12, Uord(:,1), Uord(:,2));
    U3g1 = cond_cdf(f13, p13, Uord(:,1), Uord(:,3));
    [f23g1, p23g1, logL23, k23] = fit_pair_correct(U2g1, U3g1, fams);
    total_logL = total_logL + logL23;
    total_k = total_k + k23;

    % BIC -> AICc
    AIC = -2 * total_logL + 2 * total_k;
    AICc = AIC + 2*total_k*(total_k+1)/(n - total_k - 1);
    model.AICc = AICc;

    model.order = order;
    model.Trees{1}.fams = {f12, f13};
    model.Trees{1}.params = {p12, p13};
    model.Trees{2}.fams = {f23g1};
    model.Trees{2}.params = {p23g1};
end

function [fam, param, logL, k] = fit_pair_correct(u, v, fams)
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

function U_sim = simulate_three_dim_vine_correct(N, model)
    Trees = model.Trees;
    order = model.order(:)';
    d = length(order);
    S = rand(N,d);
    U_sim_ordered = zeros(N,d);

    f12 = Trees{1}.fams{1}; p12 = Trees{1}.params{1};
    f13 = Trees{1}.fams{2}; p13 = Trees{1}.params{2};
    f23g1 = Trees{2}.fams{1}; p23g1 = Trees{2}.params{1};

    u1 = S(:,1); U_sim_ordered(:,1) = u1;
    u2 = inv_cond_cdf(f12, p12, S(:,2), u1, 'second');
    U_sim_ordered(:,2) = u2;
    u2g1 = cond_cdf(f12, p12, u1, u2);
    u3g1 = inv_cond_cdf(f23g1, p23g1, S(:,3), u2g1, 'second');
    u3 = inv_cond_cdf(f13, p13, u3g1, u1, 'second');
    U_sim_ordered(:,3) = u3;

    U_sim = zeros(N,d);
    U_sim(:, order) = U_sim_ordered;
end