function frames = frame_signal(x, frameLen, frameShift)
%FRAME_SIGNAL 简单分帧，不补零，最后不完整的一帧丢掉


    x = x(:).';  % 先转成行向量，方便索引
    numSamples = length(x);
    numFrames = floor((numSamples - frameLen) / frameShift) + 1;

    if numFrames <= 0
        frames = zeros(0, frameLen);
        return;
    end

    frames = zeros(numFrames, frameLen);
    for k = 1:numFrames
        startIdx = (k-1) * frameShift + 1;
        frames(k, :) = x(startIdx:startIdx + frameLen - 1);
    end
end
