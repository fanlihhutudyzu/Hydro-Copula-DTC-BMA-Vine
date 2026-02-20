% 基础导数函数 (dC(u,v)/du)
function h = numerich(u1, u2, fam, param)
    epsv = 1e-6;
    u1_lo = max(u1-epsv, epsv);
    u1_hi = min(u1+epsv, 1-epsv);
    c_lo = get_cdf(fam, param, u1_lo, u2);
    c_hi = get_cdf(fam, param, u1_hi, u2);
    h = (c_hi - c_lo) ./ (u1_hi - u1_lo);
    h = max(min(h, 1), 0);
end