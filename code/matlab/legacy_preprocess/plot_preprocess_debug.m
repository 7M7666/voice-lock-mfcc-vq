function plot_preprocess_debug(xRaw, xPre, xClean, fs, startSample, ...
                               energy, zcr, frameLen, frameShift, figFile)


    tRaw   = (0:length(xRaw)-1) / fs;
    tPre   = (0:length(xPre)-1) / fs;
    tClean = ((0:length(xClean)-1) + (startSample-1)) / fs;  % 加起始偏移

    figure('Position', [100 100 900 700]);

    % (1) 原始波形
    subplot(3,1,1);
    plot(tRaw, xRaw);
    title('原始语音波形');
    xlabel('Time (s)');
    ylabel('Amplitude');

    % (2) 预加重 + 去静音后波形
    subplot(3,1,2);
    plot(tPre, xPre, 'Color', [0.3 0.6 1.0]); hold on;
    plot(tClean, xClean, 'Color', [1.0 0.5 0.0]);
    legend({'预加重', '去静音后（归一化）'});
    title('预加重与端点检测结果对比');
    xlabel('Time (s)');

    % (3) 能量 & ZCR
    subplot(3,1,3);
    if ~isempty(energy) && ~isempty(zcr)
        numFrames = length(energy);
        frameTimes = ((0:numFrames-1) * frameShift + frameLen/2) / fs;

        E_norm = energy / (max(energy) + 1e-9);
        Z_norm = zcr    / (max(zcr)    + 1e-9);

        plot(frameTimes, E_norm); hold on;
        plot(frameTimes, Z_norm);
        legend({'归一化短时能量', '归一化过零率'});
        title('短时能量与过零率（用于双门限端点检测）');
        xlabel('Time (s)');
    else
        text(0.5, 0.5, '帧过短，未计算能量/ZCR', 'HorizontalAlignment', 'center');
    end

    % 保存图片
    if nargin >= 10 && ~isempty(figFile)
        [figDir,~,~] = fileparts(figFile);
        if ~exist(figDir, 'dir')
            mkdir(figDir);
        end
        saveas(gcf, figFile);
        close(gcf);
    end
end
