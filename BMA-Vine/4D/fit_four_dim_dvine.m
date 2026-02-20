function model = fit_four_dim_dvine(U, order, fams)
    n = size(U,1);
    U = U(:, order); % 1-2-3-4
    
    % Tree 1: (1,2), (2,3), (3,4)
    [f12,p12,L12,k12] = fit_pair(U(:,1), U(:,2), fams);
    [f23,p23,L23,k23] = fit_pair(U(:,2), U(:,3), fams);
    [f34,p34,L34,k34] = fit_pair(U(:,3), U(:,4), fams);
    
    % Tree 2 Inputs
    h1g2 = cond_prob(U(:,1), U(:,2), f12, p12);
    h3g2 = cond_prob(U(:,3), U(:,2), f23, p23);
    
    h2g3 = cond_prob(U(:,2), U(:,3), f23, p23);
    h4g3 = cond_prob(U(:,4), U(:,3), f34, p34);
    
    % Tree 2: (1,3)|2, (2,4)|3
    [f13,p13,L13,k13] = fit_pair(h1g2, h3g2, fams);
    [f24,p24,L24,k24] = fit_pair(h2g3, h4g3, fams);
    
    % Tree 3 Inputs
    h1g23 = cond_prob(h1g2, h3g2, f13, p13);
    h4g23 = cond_prob(h4g3, h2g3, f24, p24);
    
    % Tree 3: (1,4)|23
    [f14,p14,L14,k14] = fit_pair(h1g23, h4g23, fams);
    
    total_L = L12+L23+L34 + L13+L24 + L14;
    total_k = k12+k23+k34 + k13+k24 + k14;
    model.BIC = -2*total_L + total_k*log(n);
    model.order = order;
    
    % --- 修正点开始：显式赋值以确保是标量结构体 ---
    model.T1.f = {f12, f23, f34};
    model.T1.p = {p12, p23, p34};
    
    model.T2.f = {f13, f24};
    model.T2.p = {p13, p24};
    
    model.T3.f = {f14};
    model.T3.p = {p14};
    % --- 修正点结束 ---
end