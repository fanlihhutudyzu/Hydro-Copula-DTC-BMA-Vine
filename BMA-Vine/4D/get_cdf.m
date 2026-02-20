function val = get_cdf(fam, param, u1, u2)
    try
        if strcmp(fam,'t') || (isstruct(param) && strcmp(param.type,'t'))
            val = copulacdf('t', [u1, u2], param.rho, param.nu);
        elseif strcmp(fam,'Gaussian') || (isstruct(param) && strcmp(param.type,'Gaussian'))
            val = copulacdf('Gaussian', [u1, u2], param.rho);
        else
            if isstruct(param), th=param.theta; else, th=param; end
            val = copulacdf(fam, [u1, u2], th);
        end
    catch
        val = u1 .* u2;
    end
end