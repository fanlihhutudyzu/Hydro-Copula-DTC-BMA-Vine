function pd = pearson3fit(x)
    m = mean(x); s = std(x); g1 = skewness(x);
    if abs(g1) < 1e-6, g1 = 1e-6; end
    k = 4/(g1^2); theta = s/sqrt(k); a = m - k*theta;
    pd.k = k; pd.theta = theta; pd.a = a;
    pd.pdf = @(xx) pearson3_pdf(xx, k, theta, a);
    pd.cdf = @(xx) pearson3_cdf(xx, k, theta, a);
    pd.icdf = @(u) a + gaminv(u, k, theta);
end

function y = pearson3_pdf(x, k, theta, a)
    z = x - a; y = zeros(size(z)); idx = z > 0;
    y(idx) = gampdf(z(idx), k, theta);
end

function F = pearson3_cdf(x, k, theta, a)
    z = x - a; F = zeros(size(z)); idx = z > 0;
    F(idx) = gamcdf(z(idx), k, theta);
end