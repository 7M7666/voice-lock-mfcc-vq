function dist = feature_analysis_tool(feat1, feat2)
% FEATURE_ANALYSIS_TOOL 计算两个声纹特征矩阵之间的距离
% 
% 输入:
%   feat1 - [49 x N] 归一化特征矩阵 (Speaker A)
%   feat2 - [49 x M] 归一化特征矩阵 (Speaker B)
% 输出:
%   dist  - 标量距离值
%
% -----------------------------------------------------------

    % 1. 健壮性检查
    if isempty(feat1) || isempty(feat2)
        dist = Inf; % 无效数据距离无穷大
        return;
    end

    % -------------------------------------------------------
    % 模式 A: 基于“平均声纹向量”的欧氏距离 (推荐)
    % 原理: 将整段语音压缩为一个 49 维的平均向量，比较两者的空间距离。
    % 优点: 极快，且因为我们之前做了 Z-Score 归一化，这种方法现在非常鲁棒。
    % -------------------------------------------------------
    
    % 对时间维度 (第2维) 求均值，得到 [49 x 1] 向量
    vec1 = mean(feat1, 2);
    vec2 = mean(feat2, 2);
    
    % --- 特征加权 (可选优化) ---
    % 根据经验，MFCC (前12维) 最重要，Delta (中12维) 次之，谱特征 (后25维) 辅助
    % 这里的权重是为了让距离计算更关注“声道”特征
    % weights = [ones(12,1)*1.2;   % MFCC
    %            ones(12,1)*1.0;   % Delta
    %            ones(25,1)*0.8];  % Spectral
    % 
    % vec1 = vec1 .* sqrt(weights);
    % vec2 = vec2 .* sqrt(weights);
    % ---------------------------
    
    % 计算欧氏距离 (Euclidean Distance)
    dist = norm(vec1 - vec2);
    
    
    % -------------------------------------------------------
    % 模式 B: 动态时间规整 (DTW) (备用，计算慢)
    % 如果你想比较时序变化，可以取消下面的注释，注释掉上面的模式 A
    % -------------------------------------------------------
    %{
    % 计算两个序列的欧氏距离矩阵
    % D = pdist2(feat1', feat2', 'euclidean'); 
    
    % 简单的 DTW 累积距离 (需要 Signal Processing Toolbox)
    % [dist, ~] = dtw(feat1, feat2);
    
    % 归一化距离 (除以路径长度，防止长语音距离天然大)
    % dist = dist / (size(feat1,2) + size(feat2,2));
    %}

end