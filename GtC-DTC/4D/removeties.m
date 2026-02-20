close;clc;
load('Storm_NL_4D.mat');
data = storm_NL_4D(:,3);
eps = 0.05;
data2 = remove_ties_random_jitter(data,eps);

function X_jitter = remove_ties_random_jitter(X, eps_scale)
% remove_ties_random_jitter  使用随机扰动法消除数据中的 ties
%
% 输入：
%   X          - 原始数据，可以是向量或矩阵 (n × d)
%   eps_scale  - 扰动尺度（可选，默认 = 1e-6），表示扰动相对于数据标准差的比例
%
% 输出：
%   X_jitter   - 去除 ties 后的数据

    if nargin < 2
        eps_scale = 1e-6;   % 默认扰动比例
    end
    
    X = double(X);
    X_jitter = X;

    % 如果是一维向量，转成列向量
    if isvector(X)
        X = X(:);
        X_jitter = X_jitter(:);
    end

    [n, d] = size(X);

    for j = 1:d
        xj = X(:, j);

        % 找出存在 ties 的值
        [u, ~, idx] = unique(xj);

        % 标准差，用于缩放扰动大小
        sd = std(xj);
        if sd == 0
            sd = 1;
        end

        for k = 1:length(u)
            rows = find(idx == k);
            if numel(rows) > 1
                % 对 ties 添加微小扰动，扰动幅度取数据 sd 的 eps_scale 倍
                jitter = eps_scale * sd * randn(numel(rows), 1);
                X_jitter(rows, j) = xj(rows) + jitter;
            end
        end
    end
end
