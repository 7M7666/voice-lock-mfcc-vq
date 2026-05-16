function [startSample, endSample, energy, zcr] = endpoint_detection(x, fs, frameLen, frameShift)
% ZCR 双门限端点检测


    frames = frame_signal(x, frameLen, frameShift);

    if isempty(frames)
        startSample = 1;
        endSample   = length(x);
        energy = [];
        zcr = [];
        return;
    end

    energy = short_time_energy(frames);
    zcr    = zero_crossing_rate(frames);

    % 能量阈值（高/低）
    E_max  = max(energy);
    E_low  = 0.1 * E_max;
    E_high = 0.3 * E_max;

    % ZCR 阈值
    Z_mean = mean(zcr);
    Z_th   = 0.8 * Z_mean;

    voiced = false(size(energy));

    % 高能量帧必为语音
    voiced(energy > E_high) = true;

    % 中等能量但 ZCR 偏大：也认为是语音
    mid = (energy >= E_low) & (energy <= E_high) & (zcr > Z_th);
    voiced(mid) = true;

    if ~any(voiced)
        % 极端情况：检测不到语音，就全保留
        startSample = 1;
        endSample   = length(x);
        return;
    end

    first = find(voiced, 1, 'first');
    last  = find(voiced, 1, 'last');

    % 帧索引 -> 样本索引（1-based）
    startSample = (first-1) * frameShift + 1;
    endSample   = (last-1) * frameShift + frameLen;

    % 两侧各扩展 50ms
    pad = round(0.05 * fs);
    startSample = max(1, startSample - pad);
    endSample   = min(length(x), endSample + pad);
end
