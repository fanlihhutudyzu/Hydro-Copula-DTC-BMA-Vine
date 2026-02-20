%%%--------------- Data Generation Function (3D Scenario) --------------------%%%
function data = generate_scenario_3Ddata(scenario, n)
    d = 3;  % Three-dimensional
    % Define 3D marginal distributions (Gamma distributions with slightly different parameters)
    marg = cell(1, d);
    marg{1} = makedist('Gamma', 'a', 2.3, 'b', 2.1);
    marg{2} = makedist('Gamma', 'a', 2.5, 'b', 1.9);
    marg{3} = makedist('Gamma', 'a', 2.1, 'b', 2.2);
    switch scenario
        case 'A'  % Scenario A: t Copula (Symmetric heavy-tail correlation structure)
            % 3D correlation matrix
            R = [1    .6    .35;
                 .6  1    .4;
                 .35 .4   1];
            nu = 3;  % Degrees of freedom (smaller values indicate heavier tails)
            U = copularnd('t', R, nu, n);  % Generate t Copula samples
            
        case 'B'  % Scenario B: Mixed structure (Gumbel + Weakly correlated Gaussian)
            % Base structure: Gumbel for 1-2 (upper tail correlation), 3 is weakly correlated with 1-2
            U12 = copularnd('Gumbel', 2.2, n);  % Variable pair 1-2
            U = zeros(n, d);
            U(:, 1:2) = U12;
            
            % Generate base data for the third variable (independent of 1-2)
            U(:, 3) = rand(n, 1);
            
            % Mixed part: Introduce weakly correlated Gaussian Copula
            mix_p = 0.25;  % 25% of samples come from the weakly correlated structure
            idx_mix = rand(n, 1) < mix_p;
            if any(idx_mix)
                % Weak correlation matrix
                Rweak = [1    .15   .1;
                         .15  1    .08;
                         .1   .08   1];
                Ug = copularnd('Gaussian', Rweak, sum(idx_mix));
                U(idx_mix, :) = Ug;
            end
            
        case 'C'  % Scenario C: Mixed multi-type Copula (Asymmetric + Symmetric + Heavy-tail)
            U = zeros(n, d);
            for i = 1:n
                r = rand();  % Randomly select Copula type
                if r <= 0.5  % 50% probability: Gaussian Copula (Symmetric correlation)
                    Rg = [1    .5    .25;
                         .5  1    .35;
                         .25 .35   1];
                    U(i, :) = copularnd('Gaussian', Rg, 1);
                elseif r <= 0.8  % 30% probability: t Copula (Heavy-tail symmetric)
                    Rt = [1    .6    .3;
                         .6  1    .4;
                         .3   .4   1];
                    U(i, :) = copularnd('t', Rt, 4, 1);  % 4 degrees of freedom
                else  % 20% probability: Gumbel Copula (Upper tail correlation)
                    % Step 1: Generate Gumbel Copula for 1-2
                    U12 = copularnd('Gumbel', 2.0, 1);  % 1x2
                    % Step 2: Generate Gumbel Copula for 3|1 (Conditional dependency on U1)
                    % Use U1 as the condition to generate U3
                    u1 = U12(1);  % First variable
                    % Conditional Copula generation U3|U1=u1
                    u3_cond = copularnd('Gumbel', 1.5, 1);  % Conditional Copula parameter 1.5
                    % Combine into a 3D sample
                    U(i, :) = [U12(1), U12(2), u3_cond(1)];
                end
            end
            
        otherwise
            error('Unknown scenario');
    end
    % Convert uniform distributions back to the original data space
    data = zeros(n, d);
    for j = 1:d
        data(:, j) = icdf(marg{j}, U(:, j));
    end
end