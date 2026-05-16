function preprocess_folder(rawRoot, cleanRoot, saveFrames, saveFig, maxFigs, figStride)


    if nargin < 3, saveFrames = false; end
    if nargin < 4, saveFig = false;    end
    if nargin < 5 || isempty(maxFigs), maxFigs = Inf; end
    if nargin < 6, figStride = 1;      end

    % 列出所有 flac
    files = dir(fullfile(rawRoot, '**', '*.flac'));
    fprintf('Found %d flac files.\n', numel(files));

    figCount = 0;

    % 规范一下 rawRoot，方便算相对路径
    rawRoot = char(rawRoot);
    if rawRoot(end) ~= filesep
        rawRoot = [rawRoot filesep];
    end
    baseLen = length(rawRoot);

    for i = 1:numel(files)
        inFile = fullfile(files(i).folder, files(i).name);

        relDir = files(i).folder(baseLen+1:end);   % 相对 rawRoot 的子目录
        outDir = fullfile(cleanRoot, relDir);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end

        [~, name, ~] = fileparts(files(i).name);
        outWavFile = fullfile(outDir, [name '_clean.wav']);

        % 这一条要不要出图？
        thisSaveFig = false;
        if saveFig && (mod(i-1, figStride) == 0) && (figCount < maxFigs)
            thisSaveFig = true;
            figCount = figCount + 1;
        end

        preprocess_one_file(inFile, outWavFile, saveFrames, thisSaveFig);
    end
end
