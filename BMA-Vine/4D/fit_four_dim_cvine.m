function model = fit_four_dim_cvine(U, order, fams)
    n = size(U,1);
    U = U(:, order); % Reorder: 1 is center
    
    % Tree 1: (1,2), (1,3), (1,4)
    [f12,p12,L12,k12] = fit_pair(U(:,1), U(:,2), fams);
    [f13,p13,L13,k13] = fit_pair(U(:,1), U(:,3), fams);
    [f14,p14,L14,k14] = fit_pair(U(:,1), U(:,4), fams);
    
    % Transform to h-scale for Tree 2
    h2g1 = cond_prob(U(:,2), U(:,1), f12, p12);
    h3g1 = cond_prob(U(:,3), U(:,1), f13, p13);
    h4g1 = cond_prob(U(:,4), U(:,1), f14, p14);
    
    % Tree 2: (2,3)|1, (2,4)|1
    [f23,p23,L23,k23] = fit_pair(h2g1, h3g1, fams);
    [f24,p24,L24,k24] = fit_pair(h2g1, h4g1, fams);
    
    % Transform for Tree 3
    h3g12 = cond_prob(h3g1, h2g1, f23, p23);
    h4g12 = cond_prob(h4g1, h2g1, f24, p24);
    
    % Tree 3: (3,4)|12
    [f34,p34,L34,k34] = fit_pair(h3g12, h4g12, fams);
    
    % Store results
    total_L = L12+L13+L14 + L23+L24 + L34;
    total_k = k12+k13+k14 + k23+k24 + k34;
    model.BIC = -2*total_L + total_k*log(n);
    model.order = order;
    
    % --- 显式赋值以确保是标量结构体 ---
    model.T1.f = {f12, f13, f14};
    model.T1.p = {p12, p13, p14};
    
    model.T2.f = {f23, f24};
    model.T2.p = {p23, p24};
    
    model.T3.f = {f34};
    model.T3.p = {p34};
    
end