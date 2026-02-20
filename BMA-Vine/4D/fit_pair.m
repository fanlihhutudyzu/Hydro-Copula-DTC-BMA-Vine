function [fam, param, logL, k] = fit_pair(u, v, fams)
    bestL = -Inf; bestParam=[]; bestFam=''; bestK=0;
    for f=fams
        cfam = f{1};
        try
            switch cfam
                case 'Gaussian'
                    rho = copulafit('Gaussian',[u,v]);
                    param.rho=rho; param.type='Gaussian'; k=1;
                    pdfv = copulapdf('Gaussian',[u,v], rho);
                case 't'
                    [rho,nu] = copulafit('t',[u,v]);
                    param.rho=rho; param.nu=nu; param.type='t'; k=2;
                    pdfv = copulapdf('t',[u,v], rho, nu);
                otherwise
                    theta = copulafit(cfam, [u,v]);
                    param.theta=theta; param.type=cfam; k=1;
                    pdfv = copulapdf(cfam,[u,v], theta);
            end
            L = sum(log(max(pdfv, realmin)));
            if L>bestL, bestL=L; bestFam=cfam; bestParam=param; bestK=k; end
        catch, end
    end
    fam=bestFam; param=bestParam; logL=bestL; k=bestK;
end