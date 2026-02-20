function [gibbs_sim_data, model_info] = copula_gibbs_model(U, m)
% copula_gibbs_model  — DTC sampling
%   - Build maximum related tree (MST) from Pearson correlations
%   - Choose root = node with highest occurrence in MST edges
%   - BFS from root to determine simulation order (var_order)
%   - Fit pairwise copulas only for MST edges
%   - Simulate by sequential conditional inversion using hinv_conditional
%
% INPUTS:
%   U : n x d input data in (0,1) scale (pseudo-observations)
%   m : number of simulated samples
% OUTPUTS:
%   gibbs_sim_data : m x d simulated samples in (0,1) scale (original variable order)
%   model_info : struct including corr_mat, var_order, copula_pairs, opt_types, opt_params, root

[n,d] = size(U);
if d ~= 4
    % Still robust for general d, but your helpers were designed for d==4 originally.
    % We'll allow general d where possible, but note some later code assumes 2D pair fits.
    warning('copula_gibbs_model: expected d==4 in original script; running for d=%d', d);
end

% 1) Correlation matrix (Pearson) and list all pairs
corr_mat = corr(U, 'type', 'Pearson');
% store absolute correlations for ranking
pairs_all = [];
rho_all = [];
for i = 1:d
    for j = i+1:d
        pairs_all = [pairs_all; i, j];
        rho_all   = [rho_all; abs(corr_mat(i,j))];
    end
end

% 2) Sort pairs by descending abs(correlation)
[~, sort_idx] = sort(rho_all, 'descend');
pairs_sorted = pairs_all(sort_idx, :);

% 3) Kruskal-like selection of d-1 edges for MST using Union-Find (avoid cycles)
uf_parent = 1:d;         % union-find parent
function r = uf_find(x)
    while uf_parent(x) ~= x
        uf_parent(x) = uf_parent(uf_parent(x));
        x = uf_parent(x);
    end
    r = x;
end
function uf_union(a,b)
    ra = uf_find(a); rb = uf_find(b);
    if ra ~= rb
        uf_parent(rb) = ra;
    end
end

selected_pairs = [];
for k = 1:size(pairs_sorted,1)
    if size(selected_pairs,1) >= d-1
        break;
    end
    a = pairs_sorted(k,1);
    b = pairs_sorted(k,2);
    if uf_find(a) ~= uf_find(b)
        selected_pairs = [selected_pairs; a, b];
        uf_union(a,b);
    end
end

% If not enough edges (shouldn't happen), fallback to top d-1
if size(selected_pairs,1) < d-1
    selected_pairs = pairs_sorted(1:min(d-1,size(pairs_sorted,1)), :);
end

% 4) Determine root = node with highest occurrence in selected_pairs
nodes_list = selected_pairs(:);
uniq_nodes = unique(nodes_list);
counts = histc(nodes_list, uniq_nodes);
[~, imax] = max(counts);
root = uniq_nodes(imax);

% 5) Build adjacency list from selected_pairs and BFS to get var_order
adj = cell(1,d);
for k = 1:size(selected_pairs,1)
    i = selected_pairs(k,1);
    j = selected_pairs(k,2);
    adj{i} = unique([adj{i}, j]);
    adj{j} = unique([adj{j}, i]);
end

visited = false(1,d);
queue = root;
visited(root) = true;
order = [];
while ~isempty(queue)
    x = queue(1);
    queue(1) = [];
    order(end+1) = x; %#ok<AGROW>
    for nb = adj{x}
        if ~visited(nb)
            visited(nb) = true;
            queue(end+1) = nb; %#ok<AGROW>
        end
    end
end

% If some nodes not visited (rare), append them
if length(order) < d
    remaining = setdiff(1:d, order, 'stable');
    order = [order, remaining];
end

var_order = order; % simulation order (root first)

% 6) Fit pairwise copulas for selected_pairs
candidate_types = {'Gaussian','t','Frank','Gumbel','Clayton'};
num_pairs = size(selected_pairs,1);
opt_types = cell(1,num_pairs);
opt_params = cell(1,num_pairs);

for k = 1:num_pairs
    pair_idx = selected_pairs(k,:);
    upair = U(:, pair_idx);
    [best_type, best_params] = fit_pairwise_copula(upair, candidate_types);
    opt_types{k} = best_type;
    opt_params{k} = best_params;
end

% 7) Create quick lookup maps: for any unordered pair (i,j) find index in selected_pairs
pair_index_map = containers.Map; % key: 'i_j' (i<j) -> index k
for k = 1:num_pairs
    a = selected_pairs(k,1); b = selected_pairs(k,2);
    if a < b
        key = sprintf('%d_%d', a, b);
    else
        key = sprintf('%d_%d', b, a);
    end
    pair_index_map(key) = k;
end

% 8) Build parent mapping: for each non-root node find its parent
parent = zeros(1,d); % parent(child) = parent node index
% Use var_order to decide parent: when exploring adjacency, first encountered neighbor in BFS is parent
% We'll walk BFS order and set parent for unparented neighbors
visited = false(1,d);
visited(root) = true;
queue = root;
while ~isempty(queue)
    x = queue(1); queue(1) = [];
    for nb = adj{x}
        if ~visited(nb)
            visited(nb) = true;
            parent(nb) = x;   % x is parent of nb
            queue(end+1) = nb; %#ok<AGROW>
        end
    end
end

% 9) Simulate m samples
Usim = zeros(m, d); % in original variable indexing

% Pre-generate independent uniforms for each node and sample (one w per variable per sample)
W = rand(m, d);

% Root simulated directly
Usim(:, root) = W(:, root);

% For simulation order, ensure root is first in var_order; then iterate through var_order
for idx = 1:length(var_order)
    v = var_order(idx);
    if v == root
        continue;
    end
    p = parent(v);
    if p == 0
        error('No parent found for node %d. MST/parent mapping failed.', v);
    end

    % find pair index and determine whether stored pair has parent at first or second position
    if p < v
        key = sprintf('%d_%d', p, v);
        idx_pair = pair_index_map(key);
        stored_pair = selected_pairs(idx_pair,:); % should be [p v] or [v p]
        if stored_pair(1) == p && stored_pair(2) == v
            idx_in_pair_cond = 1; % pair stored as (parent, child) => conditioning var is 1st
        else
            idx_in_pair_cond = 2; % pair stored reversed
        end
    else
        key = sprintf('%d_%d', v, p);
        idx_pair = pair_index_map(key);
        stored_pair = selected_pairs(idx_pair,:);
        if stored_pair(1) == p && stored_pair(2) == v
            idx_in_pair_cond = 1;
        else
            idx_in_pair_cond = 2;
        end
    end

    typ = opt_types{idx_pair};
    params = opt_params{idx_pair};

    % For each sample, invert conditional using hinv_conditional
    for s_idx = 1:m
        u_cond = Usim(s_idx, p);  % value of conditioning var
        w = W(s_idx, v);          % independent uniform for this node
        try
            v_u = hinv_conditional(typ, params, u_cond, w, idx_in_pair_cond);
        catch ME
            warning('hinv_conditional failed for pair (%d,%d) type %s: %s. Using independent uniform.', p, v, typ, ME.message);
            v_u = w;
        end
        % safety clipping
        Usim(s_idx, v) = min(max(v_u, 1e-12), 1-1e-12);
    end
end

gibbs_sim_data = Usim;

% Package model_info
model_info = struct();
model_info.corr_mat = corr_mat;
model_info.var_order = var_order;
model_info.copula_pairs = selected_pairs;
model_info.opt_types = {opt_types{:}};   % cell array
model_info.opt_params = {opt_params{:}};
model_info.root = root;

end


%% ---------- Helper: get_copula_pairs_and_order ----------
function v = hinv_conditional(type, params, u_cond, w, idx_in_pair_cond)
% Solve for v in (0,1): P(V<=v|U=u_cond) = w
% The pair is C(U,V) or C(V,U), where U is the conditioning variable (u_cond)
% idx_in_pair_cond = 1 if the pair is C(U,V) and U is the 1st variable
% idx_in_pair_cond = 2 if the pair is C(V,U) and U is the 2nd variable

if idx_in_pair_cond == 1 % Pair is C(U,V). Conditional is P(V|U) = dC(u_cond, v) / d u_cond
    % We are inverting h2|1 for the pair C(U,V).
    
    if strcmpi(type,'Gaussian')
        rho = params.R(1,2);
        v = hinv_gaussian(w, u_cond, rho); 
    elseif strcmpi(type,'t')
        rho = params.R(1,2);
        nu = params.nu;
        v = hinv_t(w, u_cond, rho, nu); 
    else
        % Numerical inversion: solve F(v) = dC(u_cond, v) / d u_cond - w = 0
        % w_r_t = 1 means partial derivative w.r.t the first argument (u_cond)
        f = @(v) numeric_partial_h(type, params, u_cond, v, 1) - w;
        v = hinv_numeric(f);
    end
    
elseif idx_in_pair_cond == 2 % Pair is C(V,U). Conditional is P(V|U) = dC(v, u_cond) / d u_cond
    % We are inverting h1|2 for the pair C(V,U).
    
    if strcmpi(type,'Gaussian')
        rho = params.R(1,2);
        % For C(V,U), P(V|U) = dC(v, u)/du. Due to Gaussian symmetry, this is the same inversion.
        v = hinv_gaussian(w, u_cond, rho); 
    elseif strcmpi(type,'t')
        rho = params.R(1,2);
        nu = params.nu;
        v = hinv_t(w, u_cond, rho, nu);
    else
        % Numerical inversion: solve F(v) = dC(v, u_cond) / d u_cond - w = 0
        % w_r_t = 2 means partial derivative w.r.t the second argument (u_cond)
        f = @(v) numeric_partial_h(type, params, v, u_cond, 2) - w;
        v = hinv_numeric(f);
    end
    
else
    error('Invalid conditioning index.');
end

end

%% ---------- Helper: hinv_numeric (numeric inversion using copulacdf partial) ----------
function u_j = hinv_numeric(f)
% Solve for root of f(v) = 0 using fzero/bisection
% The function f must be passed in, as it depends on the conditional.

% initial bracket
lb = 1e-8; ub = 1-1e-8;
% try starting guess at 0.5
x0 = 0.5; 
% ensure sign change - if not, expand by searching
try
    if f(lb)*f(ub) > 0
        % try some grid points to find bracket
        grid = linspace(lb,ub,9);
        vals = arrayfun(f, grid);
        % Find the first sign change
        idx = find(vals(1:end-1).*vals(2:end) < 0, 1);
        if isempty(idx)
             % If no sign change, use the value closest to zero for initial guess
             [~, closest_idx] = min(abs(vals));
             x0 = grid(closest_idx);
        else
            lb = grid(idx);
            ub = grid(idx+1);
            x0 = (lb+ub)/2;
        end
    end
    % use fzero with initial guess
    u_j = fzero(f, x0);
catch
    % fallback to bisection
    u_j = bisection(f, lb, ub, 1e-8, 100);
end
% clamp
u_j = min(max(u_j, 1e-12), 1-1e-12);
end


%% ---------- Helper: numeric_partial_h (central difference using copulacdf) ----------
function h = numeric_partial_h(type, params, u, v, w_r_t)
% compute partial derivative of C(u,v) w.r.t the w_r_t variable numerically
% w_r_t = 1: dC(u,v)/du (or h2|1(v|u))
% w_r_t = 2: dC(u,v)/dv (or h1|2(u|v))

epsu = 1e-6;

if w_r_t == 1 % w.r.t u
    u1 = min(max(u+epsu, 0), 1);
    u0 = min(max(u-epsu, 0), 1);
    uu1 = [u1 v]; uu0 = [u0 v];
    denom = 2*epsu;
elseif w_r_t == 2 % w.r.t v
    v1 = min(max(v+epsu, 0), 1);
    v0 = min(max(v-epsu, 0), 1);
    uu1 = [u v1]; uu0 = [u v0];
    denom = 2*epsu;
else
    error('w_r_t must be 1 or 2.');
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
    % fallback: finite difference with larger eps
    epsu = 1e-4;
    if w_r_t == 1
        u1 = min(max(u+epsu, 0), 1);
        u0 = min(max(u-epsu, 0), 1);
        uu1 = [u1 v]; uu0 = [u0 v];
    else % w_r_t == 2
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

%% ---------- Helper: fit_pairwise_copula ----------
function [best_type, best_params] = fit_pairwise_copula(u_pair, candidate_types)
% Fit a set of bivariate copulas to the 2-column u_pair, return best by BIC.
n = size(u_pair,1);
best_BIC = Inf; best_type = ''; best_params = [];
for t = 1:length(candidate_types)
    type = candidate_types{t};
    try
        if strcmpi(type,'t')
            [Rhat, nuhat] = copulafit('t', u_pair);
            ll = sum(log(copulapdf('t', u_pair, Rhat, nuhat)));
            k = 1 + 1; % correlation (single rho) + nu. In 2D correlation reduces to single param
            % we interpret correlation param count as 1
            params = struct('R', Rhat, 'nu', nuhat);
        elseif strcmpi(type,'Gaussian')
            Rhat = copulafit('Gaussian', u_pair);
            ll = sum(log(copulapdf('Gaussian', u_pair, Rhat)));
            k = 1; % single rho
            params = struct('R', Rhat);
        else
            % Archimedean-type (Frank, Gumbel, Clayton) copulafit returns theta
            theta = copulafit(type, u_pair);
            ll = sum(log(copulapdf(type, u_pair, theta)));
            k = 1; % single parameter theta
            params = struct('theta', theta);
        end
        BIC = -2*ll + k*log(n);
        if BIC < best_BIC
            best_BIC = BIC;
            best_type = type;
            best_params = params;
        end
    catch ME
        warning('Failed fitting %s pair copula: %s', type, ME.message);
    end
end
% In case all fail, fallback to Gaussian estimated by rank correlation
if isempty(best_type)
    Rhat = copulafit('Gaussian', u_pair);
    best_type = 'Gaussian';
    best_params = struct('R', Rhat);
end
end


%% ---------- Helper: hinv_gaussian (analytical inverse) ----------
function v = hinv_gaussian(w, u, rho)
% For Gaussian copula with correlation rho:
% h2|1(v|u) = dC(u,v)/du. Inverse of this conditional.
% Solve w = h => Phi^{-1}(v) = rho*z_u + sqrt(1-rho^2) * Phi^{-1}(w)
z_u = norminv(u);
z_w = norminv(w);
z_v = rho * z_u + sqrt(max(0,1-rho^2)) * z_w;
v = normcdf(z_v);
end

%% ---------- Helper: hinv_t (analytical inverse for t-copula conditional) ----------
function v = hinv_t(w, u, rho, nu)
% For bivariate t copula, inverse of h2|1(v|u) = dC(u,v)/du.
z_u = tinv(u, nu);  % quantile of t_nu
% conditional: mean_c = rho*z_u; scale_c = sqrt((nu+z_u^2)/(nu+1) * (1-rho^2))
mean_c = rho * z_u;
scale_c = sqrt( (nu + z_u.^2) / (nu + 1) * (1 - rho^2) );
% we need v such that w = Ft_{nu+1}((t_v - mean_c)/scale_c)
t_w = tinv(w, nu+1);
t_v = mean_c + scale_c * t_w;
v = tcdf(t_v, nu); % map back to marginal t_nu cdf -> gives U in (0,1)
end

%% ---------- Helper: bisection  ----------
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