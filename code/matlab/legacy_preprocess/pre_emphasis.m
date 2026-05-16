function y = pre_emphasis(x, mu)
%PRE_EMPHASIS 一阶预加重滤波 y[n] = x[n] - mu * x[n-1]


    if nargin < 2
        mu = 0.95;
    end

    if isempty(x)
        y = x;
        return;
    end

    x = x(:);  % 确保是列向量
    y = [x(1); x(2:end) - mu * x(1:end-1)];
end
