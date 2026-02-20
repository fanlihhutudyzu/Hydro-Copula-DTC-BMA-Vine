%%%---------------数据生成函数（三维场景）--------------------%%%
function data = generate_scenario_3Ddata(scenario, n)
    d = 3;  % 三维
    % 定义三维边缘分布（Gamma分布，参数略有差异）
    marg = cell(1, d);
    marg{1} = makedist('Gamma', 'a', 2.3, 'b', 2.1);
    marg{2} = makedist('Gamma', 'a', 2.5, 'b', 1.9);
    marg{3} = makedist('Gamma', 'a', 2.1, 'b', 2.2);

    switch scenario
        case 'A'  % 场景A：t Copula（对称厚尾相关结构）
            % 三维相关矩阵
            R = [1    .6    .35;
                 .6  1    .4;
                 .35 .4   1];
            nu = 3;  % 自由度（较小值表示较厚的尾部）
            U = copularnd('t', R, nu, n);  % 生成t Copula样本
            
        case 'B'  % 场景B：混合结构（Gumbel + 弱相关Gaussian）
            % 基础结构：1-2用Gumbel（上尾相关），3与1-2弱相关
            U12 = copularnd('Gumbel', 2.2, n);  % 1-2变量对
            U = zeros(n, d);
            U(:, 1:2) = U12;
            
            % 生成第三变量的基础数据（与1-2独立）
            U(:, 3) = rand(n, 1);
            
            % 混合部分：引入弱相关的Gaussian Copula
            mix_p = 0.25;  % 25%的样本来自弱相关结构
            idx_mix = rand(n, 1) < mix_p;
            if any(idx_mix)
                % 弱相关矩阵
                Rweak = [1    .15   .1;
                         .15  1    .08;
                         .1   .08   1];
                Ug = copularnd('Gaussian', Rweak, sum(idx_mix));
                U(idx_mix, :) = Ug;
            end
            
        case 'C'  % 场景C：混合多类型Copula（非对称+对称+厚尾）
            U = zeros(n, d);
            for i = 1:n
                r = rand();  % 随机选择Copula类型
                if r <= 0.5  % 50%概率：Gaussian Copula（对称相关）
                    Rg = [1    .5    .25;
                         .5  1    .35;
                         .25 .35   1];
                    U(i, :) = copularnd('Gaussian', Rg, 1);
                elseif r <= 0.8  % 30%概率：t Copula（厚尾对称）
                    Rt = [1    .6    .3;
                         .6  1    .4;
                         .3   .4   1];
                    U(i, :) = copularnd('t', Rt, 4, 1);  % 自由度4
                else  % 20%概率：Gumbel Copula（上尾相关）
                    % 第一步：生成1-2的Gumbel Copula
                    U12 = copularnd('Gumbel', 2.0, 1);  % 1×2
                    % 第二步：生成3|1的Gumbel Copula（条件依赖于U1）
                    % 使用U1作为条件，生成U3
                    u1 = U12(1);  % 第一个变量
                    % 条件Copula生成U3|U1=u1
                    u3_cond = copularnd('Gumbel', 1.5, 1);  % 条件Copula参数1.5
                    % 组合为三维样本
                    U(i, :) = [U12(1), U12(2), u3_cond(1)];
                end
            end
            
        otherwise
            error('Unknown scenario');
    end

    % 将均匀分布转换为原始数据空间
    data = zeros(n, d);
    for j = 1:d
        data(:, j) = icdf(marg{j}, U(:, j));
    end
end