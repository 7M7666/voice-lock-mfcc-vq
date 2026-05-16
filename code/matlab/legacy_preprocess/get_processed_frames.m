function [framesHam, fs] = get_processed_frames(inFile)
% GET_PROCESSED_FRAMES 预处理核心模块 (整合版)
% 功能: 读取 -> 统一采样率 -> 预加重 -> VAD端点检测 -> 归一化 -> 分帧 -> 加窗
% 返回: framesHam (M帧 x 256点), fs (16000)

    % --- 1. 参数设置 (保持与开题报告一致) ---
    frameLen   = 256;    
    frameShift = 128;    
    mu         = 0.95;   
    targetFs   = 16000;  

    % --- 2. 读取音频 ---
    try
        [x, fs] = audioread(inFile);
    catch ME
        error('无法读取文件 %s: %s', inFile, ME.message);
    end
    x = x(:); % 转列向量

    % --- 3. 重采样 (统一到 16kHz) ---
    if fs ~= targetFs
        x = resample(x, targetFs, fs);
        fs = targetFs;
    end

    % --- 4. 预加重 ---
    % 依赖: pre_emphasis.m
    xPre = pre_emphasis(x, mu);

    % --- 5. 端点检测 (VAD) ---
    % 依赖: endpoint_detection.m
    [startSample, endSample, ~, ~] = endpoint_detection(xPre, fs, frameLen, frameShift);
    
    % 截取有效语音段
    xVoiced = xPre(startSample:endSample);

    % 判空保护 (防止纯静音文件报错)
    if isempty(xVoiced) || length(xVoiced) < frameLen
        warning('文件 %s 有效语音过短，跳过。', inFile);
        framesHam = [];
        return;
    end

    % --- 6. 归一化 (消除音量差异) ---
    maxAbs = max(abs(xVoiced)) + 1e-9;
    xNorm  = xVoiced / maxAbs;

    % --- 7. 分帧 ---
    % 依赖: frame_signal.m
    frames = frame_signal(xNorm, frameLen, frameShift);

    % --- 8. 加汉明窗 ---
    if ~isempty(frames)
        win = hamming(frameLen, 'periodic').'; % 行向量
        % 广播乘法: 每一帧都乘以窗函数
        framesHam = frames .* repmat(win, size(frames,1), 1);
    else
        framesHam = [];
    end
end