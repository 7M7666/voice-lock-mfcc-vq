function preprocess_one_file(inFile, outWavFile, saveFrames, saveFig)


    if nargin < 3
        saveFrames = false;
    end
    if nargin < 4
        saveFig = false;
    end

    frameLen   = 256;
    frameShift = 128;
    mu         = 0.95;
    targetFs   = 16000;

    % 1. 读取原始语音
    [x, fs] = audioread(inFile);
    if fs ~= targetFs
        x = resample(x, targetFs, fs);
        fs = targetFs;
    end

    x = x(:);  % 保证列向量

    % 2. 预加重
    xPre = pre_emphasis(x, mu);

    % 3. 端点检测
    [startSample, endSample, energy, zcr] = endpoint_detection(xPre, fs, frameLen, frameShift);
    xVoiced = xPre(startSample:endSample);

    % 4. 归一化
    maxAbs = max(abs(xVoiced)) + 1e-9;
    xNorm  = xVoiced / maxAbs;

    % 5. 分帧 + 汉明窗
    frames = frame_signal(xNorm, frameLen, frameShift);
    if ~isempty(frames)
        win = hamming(frameLen, 'periodic').';   % 1 x frameLen
        framesHam = frames .* repmat(win, size(frames,1), 1);
    else
        framesHam = frames;
    end

    % 6. 保存 clean wav
    outDir = fileparts(outWavFile);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    audiowrite(outWavFile, xNorm, fs);

    % 7. 可选：保存帧矩阵
    if saveFrames
        [~, name, ~] = fileparts(outWavFile);
        framesFile = fullfile(outDir, [name '_frames.mat']);
        frames = framesHam; %#ok<NASGU>
        save(framesFile, 'frames');
    end

    % 8. 可选：保存对比图
    if saveFig
        [~, name, ~] = fileparts(outWavFile);
        figFile = fullfile(outDir, [name '_preprocess.png']);
        plot_preprocess_debug(x, xPre, xNorm, fs, startSample, ...
                              energy, zcr, frameLen, frameShift, figFile);
    end

    fprintf('[OK] %s -> %s, frames: %d\n', inFile, outWavFile, size(framesHam,1));
end
