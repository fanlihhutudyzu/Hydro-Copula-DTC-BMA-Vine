function U_sim = simulate_four_dim_cvine_fixed(N, model)
    order = model.order;
    d = length(order);
    S = rand(N,d);
    X = zeros(N,d); % X corresponds to ordered variables
    
    % Tree 1 params
    T1 = model.T1;
    T2 = model.T2;
    T3 = model.T3;
    
    % Sample 1 (Center)
    w1 = S(:,1);
    X(:,1) = w1;
    
    % Sample 2
    % w2 given w1. Pair 12.
    % u = inv_cond_prob(target, cond)
    X(:,2) = inv_cond_prob(S(:,2), w1, T1.f{1}, T1.p{1});
    
    % Prepare for 3: h(2|1)
    h2g1 = cond_prob(X(:,2), w1, T1.f{1}, T1.p{1});
    
    % Sample 3
    % Need h(3|1). Tree 2 Pair (2,3)|1. 
    % We have h2g1. We generate h3g1 given h2g1.
    h3g1 = inv_cond_prob(S(:,3), h2g1, T2.f{1}, T2.p{1});
    % Now recover X3 from h(3|1) using Pair 13
    X(:,3) = inv_cond_prob(h3g1, w1, T1.f{2}, T1.p{2});
    
    % Prepare for 4
    h3g1_actual = cond_prob(X(:,3), w1, T1.f{2}, T1.p{2}); % Recalc for precision
    h3g12 = cond_prob(h3g1_actual, h2g1, T2.f{1}, T2.p{1});
    
    % Sample 4
    % Tree 3: Generate h(4|12) given h(3|12). Pair (3,4)|12
    h4g12 = inv_cond_prob(S(:,4), h3g12, T3.f{1}, T3.p{1});
    
    % Tree 2: Recover h(4|1) from h(4|12). Pair (2,4)|1.
    % We used h2g1 as condition in fit.
    h4g1 = inv_cond_prob(h4g12, h2g1, T2.f{2}, T2.p{2});
    
    % Tree 1: Recover X4 from h(4|1). Pair 14.
    X(:,4) = inv_cond_prob(h4g1, w1, T1.f{3}, T1.p{3});
    
    U_sim = zeros(N,d);
    U_sim(:,order) = X;
end