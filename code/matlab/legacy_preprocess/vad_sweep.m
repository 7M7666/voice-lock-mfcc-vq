function vad_sweep()

    root = 'W:\LiWanlin\learnings\lp\开放实验\preprocess2.0\Deeply Korean Read Speech Corpus';

    envs = { ...
        struct('dir','AnechoicChamber','label','Anechoic chamber (no reverb)'), ...
        struct('dir','AirbnbStudio','label','Studio apartment (moderate reverb)'), ...
        struct('dir','DanceStudio','label','Dance studio (high reverb)') ...
    };

    max_files_per_env = 200;

    try
        set(groot,'defaultAxesFontName','Microsoft YaHei');
        set(groot,'defaultTextFontName','Microsoft YaHei');
    catch
    end

    base.eHigh = 0.30;
    base.r     = 0.50;   % eLow = r * eHigh
    base.zLow  = 0.10;
    base.zHigh = 0.15;

    grid_eHigh = 0.10:0.05:0.60;
    grid_r     = 0.30:0.05:0.80;
    grid_zLow  = 0.05:0.02:0.20;
    grid_zHigh = 0.05:0.02:0.25;

    scan_and_plot(envs, root, max_files_per_env, 'eHigh', grid_eHigh, base);
    scan_and_plot(envs, root, max_files_per_env, 'r',     grid_r,     base);
    scan_and_plot(envs, root, max_files_per_env, 'zLow',  grid_zLow,  base);
    scan_and_plot(envs, root, max_files_per_env, 'zHigh', grid_zHigh, base);

end


function scan_and_plot(envs, root, maxN, whichParam, gridVals, base)

    nEnv  = numel(envs);
    nGrid = numel(gridVals);

    kept_ratio = NaN(nEnv, nGrid);
    speech_frame_ratio = NaN(nEnv, nGrid);

    fprintf('\n扫描参数: %s\n', whichParam);

    for e = 1:nEnv
        env_dir = fullfile(root, envs{e}.dir);
        wavs = dir(fullfile(env_dir, '*.wav'));
        if isempty(wavs)
            warning('%s 下没有 wav', env_dir);
            continue;
        end
        wavs = wavs(1:min(maxN, numel(wavs)));

        fprintf('[%s] files=%d\n', envs{e}.label, numel(wavs));

        for k = 1:nGrid

            p = base;
            p.(whichParam) = gridVals(k);

            eHigh = p.eHigh;
            eLow  = p.r * p.eHigh;
            zLow  = p.zLow;
            zHigh = p.zHigh;

            ratios_keep  = [];
            ratios_frame = [];

            for i = 1:numel(wavs)
                inFile = fullfile(wavs(i).folder, wavs(i).name);

                info = audioinfo(inFile);
                dur_raw = info.TotalSamples / info.SampleRate;

                [framesHam, fs, s, t, energy, zcr] = get_processed_frames_vad_sweep(inFile, eHigh, eLow, zHigh, zLow);

                if isempty(framesHam) || isempty(s) || isempty(t) || isempty(energy) || isempty(zcr)
                    continue;
                end

                dur_keep = (t - s + 1) / fs;
                ratios_keep(end+1) = dur_keep / dur_raw;

                eNorm = energy ./ max(energy + eps);
                zNorm = zcr    ./ max(zcr    + eps);
                vad_mask = (eNorm > eHigh) & (zNorm < zLow);
                ratios_frame(end+1) = mean(vad_mask);
            end

            if ~isempty(ratios_keep)
                kept_ratio(e,k) = mean(ratios_keep);
            end
            if ~isempty(ratios_frame)
                speech_frame_ratio(e,k) = mean(ratios_frame);
            end
        end
    end

    % 图1：保留时长比例
    figure; hold on;
    for e = 1:nEnv
        plot(gridVals, kept_ratio(e,:), '-o', 'DisplayName', envs{e}.label);
    end
    xlabel(xlabel_for(whichParam));
    ylabel('平均保留时长比例');
    title(sprintf('VAD阈值扫描：%s（保留时长比例）', whichParam));
    grid on;
    legend('Location','best');

    % 图2：语音帧占比
    figure; hold on;
    for e = 1:nEnv
        plot(gridVals, speech_frame_ratio(e,:), '-o', 'DisplayName', envs{e}.label);
    end
    xlabel(xlabel_for(whichParam));
    ylabel('平均语音帧占比');
    title(sprintf('VAD阈值扫描：%s（语音帧占比）', whichParam));
    grid on;
    legend('Location','best');

end


function s = xlabel_for(whichParam)
    switch whichParam
        case 'eHigh'
            s = '能量高阈值 eHigh';
        case 'r'
            s = '比例 r = eLow/eHigh';
        case 'zLow'
            s = '过零率低阈值 zLow';
        case 'zHigh'
            s = '过零率高阈值 zHigh';
        otherwise
            s = whichParam;
    end
end
