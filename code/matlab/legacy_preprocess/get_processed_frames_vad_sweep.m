function [framesHam, fs, startSample, endSample, energy, zcr] = get_processed_frames_vad_sweep(inFile, eHigh, eLow, zHigh, zLow)

    frameLen   = 256;
    frameShift = 128;
    mu         = 0.95;
    targetFs   = 16000;

    if nargin < 2, eHigh = 0.30; end
    if nargin < 3, eLow  = 0.15; end
    if nargin < 4, zHigh = 0.15; end
    if nargin < 5, zLow  = 0.10; end

    try
        [x, fs] = audioread(inFile);
    catch ME
        error('无法读取文件 %s: %s', inFile, ME.message);
    end

    if size(x,2) > 1
        x = mean(x, 2);
    end

    if fs ~= targetFs
        x = resample(x, targetFs, fs);
        fs = targetFs;
    end

    if isempty(x)
        warning('文件 %s 为空音频。', inFile);
        framesHam   = [];
        startSample = [];
        endSample   = [];
        energy      = [];
        zcr         = [];
        return;
    end

    xPre = pre_emphasis(x, mu);

    [startSample, endSample, energy, zcr] = endpoint_detection( ...
        xPre, fs, frameLen, frameShift, eHigh, eLow, zHigh, zLow);

    xVoiced = xPre(startSample:endSample);

    if isempty(xVoiced) || length(xVoiced) < frameLen
        warning('文件 %s 有效语音过短，跳过。', inFile);
        framesHam   = [];
        startSample = [];
        endSample   = [];
        energy      = [];
        zcr         = [];
        return;
    end

    maxAbs = max(abs(xVoiced));
    if maxAbs < eps
        warning('文件 %s 有效语音接近静音。', inFile);
        framesHam   = [];
        startSample = [];
        endSample   = [];
        energy      = [];
        zcr         = [];
        return;
    end

    xNorm = xVoiced / maxAbs;

    frames = frame_signal(xNorm, frameLen, frameShift);

    if ~isempty(frames)
        win = hamming(frameLen, 'periodic').';
        framesHam = frames .* repmat(win, size(frames,1), 1);
    else
        framesHam = [];
    end
end


function y = pre_emphasis(x, mu)
    if nargin < 2
        mu = 0.95;
    end

    if isempty(x)
        y = x;
        return;
    end

    x = x(:);
    y = [x(1); x(2:end) - mu * x(1:end-1)];
end


function [startSample, endSample, energy, zcr] = endpoint_detection(x, fs, frameLen, frameShift, eHigh, eLow, zHigh, zLow)

    if nargin < 5
        eHigh = 0.30;
        eLow  = 0.15;
        zHigh = 0.15;
        zLow  = 0.10;
    end

    frames = frame_signal(x, frameLen, frameShift);

    if isempty(frames)
        startSample = 1;
        endSample   = length(x);
        energy      = [];
        zcr         = [];
        return;
    end

    energy = short_time_energy(frames);
    zcr    = zero_crossing_rate(frames);

    eNorm = energy ./ max(energy + eps);
    zNorm = zcr    ./ max(zcr    + eps);

    N = length(eNorm);
    voiced = false(N,1);
    state = 0;

    for i = 1:N
        switch state
            case 0
                if eNorm(i) > eHigh
                    state = 2;
                    voiced(i) = true;
                elseif eNorm(i) > eLow
                    state = 1;
                    voiced(i) = true;
                end

            case 1
                voiced(i) = true;
                if eNorm(i) < eLow && zNorm(i) > zHigh
                    state = 0;
                elseif eNorm(i) > eHigh
                    state = 2;
                end

            case 2
                voiced(i) = true;
                if eNorm(i) < eLow && zNorm(i) > zLow
                    state = 0;
                end
        end
    end

    if ~any(voiced)
        startSample = 1;
        endSample   = length(x);
        return;
    end

    first = find(voiced, 1, 'first');
    last  = find(voiced, 1, 'last');

    startSample = (first-1) * frameShift + 1;
    endSample   = (last-1)  * frameShift + frameLen;

    pad = round(0.05 * fs);
    startSample = max(1,         startSample - pad);
    endSample   = min(length(x), endSample   + pad);
end


function E = short_time_energy(frames)
    E = sum(frames.^2, 2);
end


function zcr = zero_crossing_rate(frames)
    signs = sign(frames);
    signs(signs == 0) = 1;
    diffSign = abs(diff(signs, 1, 2));
    zcr = mean(diffSign, 2) / 2.0;
end


function frames = frame_signal(x, frameLen, frameShift)
    x = x(:).';
    numSamples = length(x);
    numFrames  = floor((numSamples - frameLen) / frameShift) + 1;

    if numFrames <= 0
        frames = zeros(0, frameLen);
        return;
    end

    frames = zeros(numFrames, frameLen);
    for k = 1:numFrames
        startIdx      = (k-1) * frameShift + 1;
        frames(k, :)  = x(startIdx:startIdx + frameLen - 1);
    end
end
