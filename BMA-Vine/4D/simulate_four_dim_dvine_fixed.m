function U_sim = simulate_four_dim_dvine_fixed(N, model)
    order = model.order;
    d = length(order);
    S = rand(N,d);
    X = zeros(N,d);
    
    T1 = model.T1; T2 = model.T2; T3 = model.T3;
    
    % 1. X1
    X(:,1) = S(:,1);
    
    % 2. X2 given X1. Pair 12.
    X(:,2) = inv_cond_prob(S(:,2), X(:,1), T1.f{1}, T1.p{1});
    
    % Prep h(1|2)
    h1g2 = cond_prob(X(:,1), X(:,2), T1.f{1}, T1.p{1});
    
    % 3. X3
    % Need h(3|2). Tree 2 Pair (1,3)|2. (Fit: h1g2, h3g2)
    % We have h1g2. Generate h3g2 given h1g2.
    h3g2 = inv_cond_prob(S(:,3), h1g2, T2.f{1}, T2.p{1});
    
    % Recover X3 from h(3|2). Tree 1 Pair 23.
    X(:,3) = inv_cond_prob(h3g2, X(:,2), T1.f{2}, T1.p{2});
    
    % Prep for 4
    h3g2_new = cond_prob(X(:,3), X(:,2), T1.f{2}, T1.p{2}); 
    h1g23 = cond_prob(h1g2, h3g2_new, T2.f{1}, T2.p{1});
    
    h2g3 = cond_prob(X(:,2), X(:,3), T1.f{2}, T1.p{2});
    
    % 4. X4
    % Tree 3 Pair (1,4)|23. (Fit: h1g23, h4g23).
    % Generate h4g23 given h1g23.
    h4g23 = inv_cond_prob(S(:,4), h1g23, T3.f{1}, T3.p{1});
    
    % Tree 2 Pair (2,4)|3. (Fit: h2g3, h4g3).
    % Recover h4g3 from h4g23 given h2g3.
    h4g3 = inv_cond_prob(h4g23, h2g3, T2.f{2}, T2.p{2});
    
    % Tree 1 Pair 34. (Fit: X3, X4).
    % Recover X4 from h4g3 given X3.
    X(:,4) = inv_cond_prob(h4g3, X(:,3), T1.f{3}, T1.p{3});
    
    U_sim = zeros(N,d);
    U_sim(:,order) = X;
end