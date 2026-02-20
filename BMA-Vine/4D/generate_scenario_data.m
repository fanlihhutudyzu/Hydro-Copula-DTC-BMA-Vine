function data = generate_scenario_data(scenario, n)
    d = 4;
    marg = cell(1,d);
    marg{1} = makedist('Gamma','a',2.3,'b',2.1);
    marg{2} = makedist('Gamma','a',2.5,'b',1.9);
    marg{3} = makedist('Gamma','a',2.1,'b',2.2);
    marg{4} = makedist('Gamma','a',2.4,'b',1.8);

    switch scenario
        case 'A'
            R = [1 .6 .35 .2; .6 1 .4 .25; .35 .4 1 .35; .2 .25 .35 1];
            nu = 3;
            U = copularnd('t', R, nu, n);
        case 'B'
            U12 = copularnd('Clayton', 2.2, n);
            U34 = copularnd('Gumbel', 2.5, n);
            U = zeros(n,4);
            U(:,1:2) = U12;
            U(:,3:4) = U34;
            mix_p = 0.25;
            idx_mix = rand(n,1) < mix_p;
            if any(idx_mix)
                Rweak = eye(4); Rweak(1,3)=0.18; Rweak(3,1)=0.18; Rweak(1,4)=0.12; Rweak(4,1)=0.12; Rweak(2,3)=0.1; Rweak(3,2)=0.1;
                Ug = copularnd('Gaussian', Rweak, sum(idx_mix));
                U(idx_mix,:) = Ug;
            end
        case 'C'
            U = zeros(n,4);
            for i=1:n
                r = rand();
                if r <= 0.5
                    Rg = [1 .5 .25 .2; .5 1 .35 .25; .25 .35 1 .3; .2 .25 .3 1];
                    U(i,:) = copularnd('Gaussian', Rg, 1);
                elseif r <= 0.8
                    Rt = [1 .6 .3 .2; .6 1 .4 .25; .3 .4 1 .35; .2 .25 .35 1];
                    U(i,:) = copularnd('t', Rt, 4, 1);
                else
                    c12 = copularnd('Clayton', 2, 1);
                    c34 = copularnd('Gumbel', 2.2, 1);
                    U(i,:) = [c12(1), c12(2), c34(1), c34(2)];
                end
            end
        otherwise
            error('Unknown scenario');
    end

    data = zeros(n,d);
    for j=1:d
        data(:,j) = icdf(marg{j}, U(:,j));
    end
end