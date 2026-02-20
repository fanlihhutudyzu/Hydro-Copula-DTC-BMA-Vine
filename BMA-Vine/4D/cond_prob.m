% 计算 h(u|v) = P(U<=u | V=v) = dC(u,v)/dv
function p = cond_prob(target, condition, fam, param)
    % 这里的逻辑是：求 target 给定 condition 的条件概率
    % 对应数学上的 dC(target, condition) / d(condition)
    % 由于 Copula 对称性 (Gaussian, t, Clayton, Gumbel, Frank)，
    % dC(u,v)/dv 等价于 dC(v,u)/dv (如果把v放在第一个位置求导)
    % 使用 numerich(condition, target) 即可
    p = numerich(condition, target, fam, param);
end