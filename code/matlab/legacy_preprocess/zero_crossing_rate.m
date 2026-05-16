function zcr = zero_crossing_rate(frames)
%ZERO_CROSSING_RATE 帧过零率：符号变化次数 / (2 * 帧长)

    signs = sign(frames);
    signs(signs == 0) = 1;
    diffSign = abs(diff(signs, 1, 2));   % 在列方向做差
    zcr = mean(diffSign, 2) / 2.0;
end
