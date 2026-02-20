% 计算条件分布的逆函数：已知 v 和 P(U<=u|V=v)=p，求 u
function u = inv_cond_prob(p, condition, fam, param)
    % 求解 numerich(condition, u) = p
    % 使用二分法求解
    n = length(p);
    u = zeros(n,1);
    tol = 1e-6;
    for i=1:n
        tgt = p(i);
        cond_val = condition(i);
        cond_val = max(min(cond_val, 1-eps), eps);
        tgt = max(min(tgt, 1-eps), eps);
        
        low = 0; high = 1;
        for iter=1:50
            mid = 0.5*(low+high);
            % 计算 P(Mid | Cond)
            val = numerich(cond_val, mid, fam, param);
            
            % 判断单调性 (基于 Copula 性质，正相关时随mid增加而增加)
            if iter==1
                v_hi = numerich(cond_val, 0.99, fam, param);
                v_lo = numerich(cond_val, 0.01, fam, param);
                is_incr = (v_hi > v_lo);
            end
            
            if is_incr
                if val < tgt, low=mid; else, high=mid; end
            else
                if val < tgt, high=mid; else, low=mid; end
            end
            if (high-low)<tol, break; end
        end
        u(i) = 0.5*(low+high);
    end
end