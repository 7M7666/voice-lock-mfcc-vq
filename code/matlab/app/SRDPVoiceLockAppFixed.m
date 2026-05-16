classdef SRDPVoiceLockAppFixed < matlab.apps.AppBase

    properties (Access = public)
        UIFigure              matlab.ui.Figure
        GridLayout            matlab.ui.container.GridLayout
        HeaderPanel           matlab.ui.container.Panel
        TitleLabel            matlab.ui.control.Label
        ControlPanel          matlab.ui.container.Panel
        DemoLabel             matlab.ui.control.Label
        DemoDropDown          matlab.ui.control.DropDown
        FilterLabel           matlab.ui.control.Label
        SampleFilterDropDown  matlab.ui.control.DropDown
        AllSampleLabel        matlab.ui.control.Label
        AllSampleDropDown     matlab.ui.control.DropDown
        RunDemoBTN            matlab.ui.control.Button
        LoadBTN               matlab.ui.control.Button
        PlayBTN               matlab.ui.control.Button
        ExtractBTN            matlab.ui.control.Button
        MatchBTN              matlab.ui.control.Button
        EnvLabel              matlab.ui.control.Label
        EnvDropDown           matlab.ui.control.DropDown
        UserLabel             matlab.ui.control.Label
        TargetUserDropDown    matlab.ui.control.DropDown
        FileNameLabel         matlab.ui.control.Label
        SampleRateLabel       matlab.ui.control.Label
        SampleTypeLabel       matlab.ui.control.Label
        FeaturePanel          matlab.ui.container.Panel
        FeatureGrid           matlab.ui.container.GridLayout
        WaveformLabel         matlab.ui.control.Label
        MFCCLabel             matlab.ui.control.Label
        WaveformAxes          matlab.ui.control.UIAxes
        MFCCAxes              matlab.ui.control.UIAxes
        ResultPanel           matlab.ui.container.Panel
        ResultTitleLabel      matlab.ui.control.Label
        TextArea              matlab.ui.control.TextArea
        UITable               matlab.ui.control.Table
    end

    properties (Access = private)
        AudioPath
        AudioData
        SampleRate
        FeaturesExtracted = false
        ResultTable = table()
        ProjectRoot = ''
        SelectedResultRow = table()
        SelectedSamplePath = ''
        DemoSampleLabels = {}
        DemoSamplePaths = {}
        AllSampleLabels = {}
        AllSamplePaths = {}
    end

    methods (Access = private)

        function initializeAppState(app)
            app.ProjectRoot = app.findProjectRoot();
            app.DemoSampleLabels = {
                '正常通过 - S01 走廊 004'
                '正常通过 - S02 走廊 005'
                '错误口令拒绝 - S01 wrong 宿舍 003'
                '冒充用户拒绝 - U01 001'
                '边界样例 - S01 实验室 004'
            };
            app.DemoSamplePaths = {
                'data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav'
                'data/demo_audio/srdp_dataset/S02/corridor/S02_corridor_005.wav'
                'data/demo_audio/srdp_dataset/S01_wrong/dorm/S01_wrong_003.wav'
                'data/demo_audio/srdp_dataset/U01/dorm/U01_corridor_001.wav'
                'data/demo_audio/srdp_dataset/S01/lab/S01_lab_004.wav'
            };
            app.DemoDropDown.Items = app.DemoSampleLabels;
            app.DemoDropDown.Value = app.DemoSampleLabels{1};
            app.SelectedSamplePath = app.DemoSamplePaths{1};

            app.loadResultTable();
            app.populateSampleList();
            app.syncContextDisplay(table(), '');
            app.TextArea.Value = {
                '请选择演示样例'
                '点击“一键演示”自动加载音频、提取特征并显示识别结论'
            };
            app.TextArea.FontColor = '#1F2937';
            app.TextArea.BackgroundColor = '#FFFFFF';
        end

        function cmap = npgMfccColormap(app)
            anchors = [
                60 84 136
                255 255 255
                230 75 53
            ] / 255;
            x = [0 0.5 1];
            xi = linspace(0, 1, 256)';
            cmap = [
                interp1(x, anchors(:, 1), xi), ...
                interp1(x, anchors(:, 2), xi), ...
                interp1(x, anchors(:, 3), xi)
            ];
        end

        function featureMap = normalizeMfccForDisplay(app, coeffs)
            featureMap = coeffs';
            rowMean = mean(featureMap, 2);
            rowStd = std(featureMap, 0, 2);
            rowStd(rowStd < 1e-6) = 1;
            featureMap = bsxfun(@minus, featureMap, rowMean);
            featureMap = bsxfun(@rdivide, featureMap, rowStd);
            featureMap = max(min(featureMap, 3), -3);
        end

        function roots = appendRootAndParents(app, roots, startPath)
            current = char(startPath);
            while ~isempty(current)
                roots{end + 1, 1} = current; %#ok<AGROW>
                parent = fileparts(current);
                if isempty(parent) || strcmp(parent, current)
                    break;
                end
                current = parent;
            end
        end

        function root = findProjectRoot(app)
            appFile = mfilename('fullpath');
            if strlength(string(appFile)) > 0
                appDir = fileparts(appFile);
            else
                appDir = pwd;
            end

            candidates = {};
            candidates = app.appendRootAndParents(candidates, pwd);
            candidates = app.appendRootAndParents(candidates, appDir);
            for i = 1:numel(candidates)
                candidate = candidates{i};
                hasRunScript = isfile(fullfile(candidate, 'run.ps1'));
                hasDataDir = isfolder(fullfile(candidate, 'data', 'demo_audio'));
                if hasRunScript && hasDataDir
                    root = candidate;
                    return;
                end
            end
            root = pwd;
        end

        function resultPath = recognitionResultPath(app)
            resultPath = fullfile(app.ProjectRoot, 'results', 'tables', 'lbg_vq_recognition_results.csv');
        end

        function loadResultTable(app)
            resultPath = app.recognitionResultPath();
            if ~isfile(resultPath)
                app.ResultTable = table();
                app.TextArea.Value = {
                    '尚未找到 Python 结果表'
                    '请先在项目根目录运行: .\run.ps1 pipeline'
                };
                app.TextArea.FontColor = '#C0392B';
                app.TextArea.BackgroundColor = '#FFF5F5';
                return;
            end
            app.ResultTable = readtable(resultPath, 'TextType', 'string');
        end

        function populateSampleList(app)
            if isempty(app.ResultTable) || height(app.ResultTable) == 0
                app.AllSampleLabels = {'结果表未加载'};
                app.AllSamplePaths = {};
                app.AllSampleDropDown.Items = app.AllSampleLabels;
                app.AllSampleDropDown.Value = app.AllSampleLabels{1};
                return;
            end

            T = app.ResultTable;
            filterValue = app.SampleFilterDropDown.Value;
            splitValues = string(T.split);
            correctValues = string(T.correct);
            isCorrectValues = strcmpi(correctValues, "true");

            if strcmp(filterValue, '正常测试')
                T = T(splitValues == "test", :);
            elseif strcmp(filterValue, '错误口令')
                T = T(splitValues == "wrong_passphrase", :);
            elseif strcmp(filterValue, '冒充用户')
                T = T(splitValues == "impostor", :);
            elseif strcmp(filterValue, '边界样例')
                T = T(~isCorrectValues, :);
            end

            if height(T) == 0
                app.AllSampleLabels = {'无匹配样本'};
                app.AllSamplePaths = {};
                app.AllSampleDropDown.Items = app.AllSampleLabels;
                app.AllSampleDropDown.Value = app.AllSampleLabels{1};
                return;
            end

            labels = cell(height(T), 1);
            paths = cell(height(T), 1);
            for i = 1:height(T)
                row = T(i, :);
                labels{i} = app.makeSampleLabel(row);
                paths{i} = app.rowText(row, 'file_path');
            end

            app.AllSampleLabels = labels;
            app.AllSamplePaths = paths;
            app.AllSampleDropDown.Items = labels;
            app.AllSampleDropDown.Value = labels{1};
        end

        function label = makeSampleLabel(app, row)
            filePath = app.rowText(row, 'file_path');
            [~, fileName, fileExt] = fileparts(filePath);
            typeText = app.splitToText(app.rowText(row, 'split'), app.rowText(row, 'correct'));
            expectedText = app.decisionToText(app.rowText(row, 'expected_decision'));
            decisionText = app.decisionToText(app.rowText(row, 'decision'));
            label = [typeText, ' | ', fileName, fileExt, ' | 期望:', expectedText, ' -> ', decisionText];
        end

        function audioPath = resolveAudioPath(app, audioPath)
            audioPath = char(audioPath);
            if isfile(audioPath)
                return;
            end
            normalized = strrep(audioPath, '/', filesep);
            candidate = fullfile(app.ProjectRoot, normalized);
            if isfile(candidate)
                audioPath = candidate;
            end
        end

        function loadAudioFromPath(app, audioPath)
            audioPath = app.resolveAudioPath(audioPath);
            if ~isfile(audioPath)
                app.AudioPath = '';
                app.AudioData = [];
                app.SampleRate = [];
                app.FeaturesExtracted = false;
                uialert(app.UIFigure, ['找不到音频文件: ', audioPath], '文件缺失', 'Icon', 'error');
                return;
            end

            [y, fs] = audioread(audioPath);
            if size(y, 2) > 1
                y = mean(y, 2);
            end

            app.AudioPath = audioPath;
            app.AudioData = y;
            app.SampleRate = fs;
            app.FeaturesExtracted = false;

            [~, fileName, fileExt] = fileparts(audioPath);
            app.FileNameLabel.Text = ['文件名: ', fileName, fileExt];
            app.SampleRateLabel.Text = ['采样率: ', num2str(fs), ' Hz'];

            plot(app.WaveformAxes, (0:length(y)-1) / fs, y, 'Color', '#3C5488', 'LineWidth', 0.8);
            app.WaveformAxes.XLim = [0, length(y) / fs];
            app.WaveformAxes.Color = '#FFFFFF';
            grid(app.WaveformAxes, 'on');
            cla(app.MFCCAxes);

            row = app.findRecognitionRow(audioPath);
            app.syncContextDisplay(row, audioPath);
            if height(row) == 0
                app.UITable.Data = {};
                if ~isfile(app.recognitionResultPath())
                    app.TextArea.Value = {
                        '尚未找到 Python 结果表'
                        '请先在项目根目录运行: .\run.ps1 pipeline'
                    };
                    app.TextArea.FontColor = '#C0392B';
                    app.TextArea.BackgroundColor = '#FFF5F5';
                else
                    app.TextArea.Value = {
                        '已选择手动音频'
                        '点击“一键演示”或“运行识别”读取后端结果'
                    };
                    app.TextArea.FontColor = '#1F2937';
                    app.TextArea.BackgroundColor = '#FFFFFF';
                end
            end
        end

        function extractDisplayFeatures(app)
            if isempty(app.AudioData)
                uialert(app.UIFigure, '请先选择或加载一个音频样本。', '操作提示', 'Icon', 'warning');
                return;
            end

            displayAudio = app.AudioData;
            fs = app.SampleRate;
            if max(abs(displayAudio)) > 0
                displayAudio = displayAudio / max(abs(displayAudio));
            end

            coeffs = mfcc(displayAudio, fs);
            featureMap = app.normalizeMfccForDisplay(coeffs);
            imagesc(app.MFCCAxes, featureMap, [-3 3]);
            axis(app.MFCCAxes, 'xy');
            colormap(app.MFCCAxes, app.npgMfccColormap());
            title(app.MFCCAxes, 'MFCC 特征');
            xlabel(app.MFCCAxes, 'Frame Index');
            ylabel(app.MFCCAxes, 'MFCC Coefficient Index');
            app.MFCCAxes.Color = '#FFFFFF';

            app.FeaturesExtracted = true;
        end

        function row = findRecognitionRow(app, audioPath)
            row = table();
            app.SelectedResultRow = table();
            if isempty(app.ResultTable) || height(app.ResultTable) == 0
                app.loadResultTable();
            end
            if isempty(app.ResultTable) || height(app.ResultTable) == 0
                return;
            end

            [~, audioName, audioExt] = fileparts(audioPath);
            audioFile = string(audioName) + string(audioExt);
            filePaths = replace(string(app.ResultTable.file_path), '\', '/');
            normalizedAudioPath = replace(string(audioPath), '\', '/');
            normalizedProjectRoot = replace(string(app.ProjectRoot), '\', '/');
            if ~endsWith(normalizedProjectRoot, "/")
                normalizedProjectRoot = normalizedProjectRoot + "/";
            end

            relativeAudioPath = normalizedAudioPath;
            if startsWith(relativeAudioPath, normalizedProjectRoot)
                relativeAudioPath = extractAfter(relativeAudioPath, strlength(normalizedProjectRoot));
            end

            idx = find(filePaths == relativeAudioPath, 1, 'first');
            if isempty(idx)
                idx = find(arrayfun(@(p) endsWith(normalizedAudioPath, p), filePaths), 1, 'first');
            end
            if isempty(idx)
                idx = find(endsWith(filePaths, audioFile), 1, 'first');
            end
            if ~isempty(idx)
                row = app.ResultTable(idx, :);
                app.SelectedResultRow = row;
            end
        end

        function renderRecognitionResult(app, row)
            if isempty(row) || height(row) == 0
                if ~isfile(app.recognitionResultPath())
                    app.TextArea.Value = {
                        '尚未找到 Python 结果表'
                        '请先在项目根目录运行: .\run.ps1 pipeline'
                    };
                    app.TextArea.FontColor = '#C0392B';
                    app.TextArea.BackgroundColor = '#FFF5F5';
                    app.UITable.Data = {};
                    uialert(app.UIFigure, '找不到 results/tables/lbg_vq_recognition_results.csv。请先在项目根目录运行 .\run.ps1 pipeline。', '文件缺失', 'Icon', 'error');
                    return;
                end
                [~, audioName, audioExt] = fileparts(app.AudioPath);
                audioFile = [audioName, audioExt];
                app.TextArea.Value = {
                    '当前音频不在正式结果表'
                    '可能是训练样本或未重新运行 pipeline'
                };
                app.TextArea.FontColor = '#3C5488';
                app.TextArea.BackgroundColor = '#F4F7FB';
                app.UITable.Data = {};
                uialert(app.UIFigure, sprintf('CSV 中找不到当前音频: %s\n该音频可能是训练样本，请选择 *_004.wav、*_005.wav、wrong 或 U01 测试样本。', audioFile), '未匹配到结果', 'Icon', 'warning');
                return;
            end

            decisionText = app.rowText(row, 'decision');
            voiceDecision = app.rowText(row, 'voice_decision');
            passphrasePassed = app.rowText(row, 'passphrase_passed');
            correctText = app.rowText(row, 'correct');
            expectedText = app.rowText(row, 'expected_decision');

            isCorrect = strcmpi(correctText, 'True');
            if ~strcmp(decisionText, 'REJECT')
                mainText = ['通过：', decisionText];
                reasonText = '声纹匹配，口令通过';
                fgColor = '#007D5C';
                bgColor = '#F0FFF4';
            elseif strcmp(voiceDecision, 'REJECT')
                mainText = '拒绝';
                reasonText = '声纹未通过';
                fgColor = '#C0392B';
                bgColor = '#FFF5F5';
            elseif strcmpi(passphrasePassed, 'False')
                mainText = '拒绝';
                reasonText = '口令不匹配';
                fgColor = '#C0392B';
                bgColor = '#FFF5F5';
            else
                mainText = '拒绝';
                reasonText = '系统拒绝';
                fgColor = '#C0392B';
                bgColor = '#FFF5F5';
            end

            if ~isCorrect
                mainText = [mainText, '（边界样例）'];
                if ~strcmp(expectedText, 'REJECT') && strcmp(decisionText, 'REJECT')
                    reasonText = '注册用户被误拒，可用于说明后续优化方向';
                else
                    reasonText = '该样本未判对，可用于说明系统边界';
                end
                fgColor = '#3C5488';
                bgColor = '#F4F7FB';
            end

            app.TextArea.Value = {mainText; reasonText};
            app.TextArea.FontSize = 24;
            app.TextArea.FontWeight = 'bold';
            app.TextArea.FontColor = fgColor;
            app.TextArea.BackgroundColor = bgColor;
            app.renderDetailTable(row);
            app.syncContextDisplay(row, app.AudioPath);
        end

        function renderDetailTable(app, row)
            filePath = app.rowText(row, 'file_path');
            [~, fileName, fileExt] = fileparts(filePath);
            typeText = app.splitToText(app.rowText(row, 'split'), app.rowText(row, 'correct'));
            expectedText = app.decisionToText(app.rowText(row, 'expected_decision'));
            voiceText = app.decisionToText(app.rowText(row, 'voice_decision'));
            passText = app.passphraseToText(app.rowText(row, 'passphrase_passed'));
            decisionText = app.decisionToText(app.rowText(row, 'decision'));
            correctText = app.correctToText(app.rowText(row, 'correct'));
            voiceDistance = app.distancePairText(row, 'distance', 'threshold');
            passDistance = app.distancePairText(row, 'passphrase_distance', 'passphrase_threshold');

            app.UITable.ColumnName = {'文件名'; '类型'; '期望'; '声纹判定'; '口令验证'; '最终判定'; '是否正确'; '声纹距离'; '口令距离'};
            app.UITable.ColumnWidth = {120, 78, 62, 70, 70, 70, 62, 90, 90};
            app.UITable.Data = {
                [fileName, fileExt], typeText, expectedText, voiceText, passText, decisionText, correctText, voiceDistance, passDistance
            };
        end

        function syncContextDisplay(app, row, audioPath)
            userText = '未选择';
            envText = '未选择';
            typeText = '样本类型: 未选择';

            if ~isempty(row) && height(row) > 0
                filePath = app.rowText(row, 'file_path');
                [~, fileName, ~] = fileparts(filePath);
                userText = app.userFromFileName(fileName);
                envText = app.environmentToText(app.rowText(row, 'environment'));
                typeText = ['样本类型: ', app.splitToText(app.rowText(row, 'split'), app.rowText(row, 'correct'))];
            elseif ~isempty(audioPath)
                [~, fileName, ~] = fileparts(audioPath);
                userText = app.userFromFileName(fileName);
                envText = app.environmentFromFileName(fileName);
                typeText = '样本类型: 手动音频';
            end

            app.TargetUserDropDown.Items = {userText};
            app.TargetUserDropDown.Value = userText;
            app.TargetUserDropDown.Enable = 'off';
            app.EnvDropDown.Items = {envText};
            app.EnvDropDown.Value = envText;
            app.EnvDropDown.Enable = 'off';
            app.SampleTypeLabel.Text = typeText;
        end

        function runSelectedSample(app)
            audioPath = app.SelectedSamplePath;
            if isempty(audioPath) && ~isempty(app.AudioPath)
                audioPath = app.AudioPath;
            end
            if isempty(audioPath)
                uialert(app.UIFigure, '请先选择答辩样例、正式样本或手动音频。', '操作提示', 'Icon', 'warning');
                return;
            end

            app.UIFigure.Pointer = 'watch';
            drawnow;
            try
                app.loadAudioFromPath(audioPath);
                if isempty(app.AudioData)
                    app.UIFigure.Pointer = 'arrow';
                    return;
                end
                app.extractDisplayFeatures();
                row = app.findRecognitionRow(app.AudioPath);
                app.renderRecognitionResult(row);
            catch ME
                uialert(app.UIFigure, ['一键演示失败: ', ME.message], '系统错误', 'Icon', 'error');
            end
            app.UIFigure.Pointer = 'arrow';
        end

        function text = rowText(app, row, colName)
            text = '';
            if isempty(row) || height(row) == 0 || ~ismember(colName, row.Properties.VariableNames)
                return;
            end
            value = string(row.(colName));
            if isempty(value)
                return;
            end
            value = value(1);
            if ismissing(value)
                return;
            end
            text = char(value);
        end

        function text = splitToText(app, splitValue, correctValue)
            if strcmpi(correctValue, 'False')
                text = '边界样例';
            elseif strcmp(splitValue, 'test')
                text = '正常测试';
            elseif strcmp(splitValue, 'wrong_passphrase')
                text = '错误口令';
            elseif strcmp(splitValue, 'impostor')
                text = '冒充用户';
            else
                text = splitValue;
            end
        end

        function text = environmentToText(app, envValue)
            if strcmp(envValue, 'corridor')
                text = '走廊';
            elseif strcmp(envValue, 'dorm')
                text = '宿舍';
            elseif strcmp(envValue, 'lab')
                text = '实验室';
            else
                text = envValue;
            end
        end

        function text = environmentFromFileName(app, fileName)
            if contains(fileName, '_corridor_')
                text = '走廊';
            elseif contains(fileName, '_dorm_')
                text = '宿舍';
            elseif contains(fileName, '_lab_')
                text = '实验室';
            else
                text = '未知';
            end
        end

        function text = userFromFileName(app, fileName)
            parts = split(string(fileName), '_');
            if numel(parts) >= 2 && strcmp(parts(2), "wrong")
                text = char(parts(1) + "_wrong");
            elseif numel(parts) >= 1
                text = char(parts(1));
            else
                text = '未知';
            end
        end

        function text = decisionToText(app, decision)
            if isempty(decision)
                text = '-';
            elseif strcmp(decision, 'REJECT')
                text = '拒绝';
            else
                text = decision;
            end
        end

        function text = passphraseToText(app, passphrasePassed)
            if strcmpi(passphrasePassed, 'True')
                text = '通过';
            elseif strcmpi(passphrasePassed, 'False')
                text = '不通过';
            else
                text = '-';
            end
        end

        function text = correctToText(app, correctValue)
            if strcmpi(correctValue, 'True')
                text = '正确';
            elseif strcmpi(correctValue, 'False')
                text = '误判';
            else
                text = '-';
            end
        end

        function text = distancePairText(app, row, distCol, thresholdCol)
            distText = app.rowText(row, distCol);
            thresholdText = app.rowText(row, thresholdCol);
            distValue = str2double(distText);
            thresholdValue = str2double(thresholdText);
            if isnan(distValue) || isnan(thresholdValue)
                text = '-';
            else
                text = sprintf('%.3f / %.3f', distValue, thresholdValue);
            end
        end

        function DemoDropDownValueChanged(app, event)
            idx = find(strcmp(app.DemoSampleLabels, app.DemoDropDown.Value), 1, 'first');
            if ~isempty(idx)
                app.SelectedSamplePath = app.DemoSamplePaths{idx};
            end
        end

        function SampleFilterDropDownValueChanged(app, event)
            app.populateSampleList();
            if ~isempty(app.AllSamplePaths)
                app.SelectedSamplePath = app.AllSamplePaths{1};
            end
        end

        function AllSampleDropDownValueChanged(app, event)
            idx = find(strcmp(app.AllSampleLabels, app.AllSampleDropDown.Value), 1, 'first');
            if ~isempty(idx) && idx <= numel(app.AllSamplePaths)
                app.SelectedSamplePath = app.AllSamplePaths{idx};
            end
        end

        function RunDemoBTNButtonPushed(app, event)
            app.runSelectedSample();
        end

        function LoadBTNButtonPushed(app, event)
            [file, path] = uigetfile({'*.wav', 'WAV Audio (*.wav)'});
            if isequal(file, 0)
                return;
            end
            app.SelectedSamplePath = fullfile(path, file);
            app.loadAudioFromPath(app.SelectedSamplePath);
        end

        function PlayBTNButtonPushed(app, event)
            if isempty(app.AudioData)
                uialert(app.UIFigure, '请先导入语音文件。', '提示', 'Icon', 'warning');
                return;
            end
            player = audioplayer(app.AudioData, app.SampleRate);
            play(player);
            assignin('base', 'temp_player', player);
        end

        function ExtractBTNButtonPushed(app, event)
            try
                app.extractDisplayFeatures();
                if app.FeaturesExtracted
                    app.TextArea.Value = {
                        '声学特征已提取'
                        '点击“运行识别”读取 Python 后端结果'
                    };
                    app.TextArea.FontSize = 18;
                    app.TextArea.FontWeight = 'bold';
                    app.TextArea.FontColor = '#1F2937';
                    app.TextArea.BackgroundColor = '#FFFFFF';
                end
            catch ME
                uialert(app.UIFigure, ['特征提取失败: ', ME.message], '系统错误', 'Icon', 'error');
                app.FeaturesExtracted = false;
            end
        end

        function MatchBTNButtonPushed(app, event)
            if isempty(app.AudioPath)
                uialert(app.UIFigure, '请先选择或加载一个音频样本。', '操作提示', 'Icon', 'warning');
                return;
            end
            try
                if ~app.FeaturesExtracted
                    app.extractDisplayFeatures();
                end
                row = app.findRecognitionRow(app.AudioPath);
                app.renderRecognitionResult(row);
            catch ME
                uialert(app.UIFigure, ['读取或显示识别结果失败: ', ME.message], '错误', 'Icon', 'error');
            end
        end
    end

    methods (Access = private)

        function createComponents(app)
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [0.9569 0.9686 0.9765];
            app.UIFigure.Position = [100 100 1280 800];
            app.UIFigure.Name = 'SRDP 声音锁演示系统';

            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {280, '1x', 400};
            app.GridLayout.RowHeight = {82, '1x'};

            app.ControlPanel = uipanel(app.GridLayout);
            app.ControlPanel.Title = '演示控制台';
            app.ControlPanel.Layout.Row = 2;
            app.ControlPanel.Layout.Column = 1;

            app.DemoLabel = uilabel(app.ControlPanel);
            app.DemoLabel.Position = [24 625 220 22];
            app.DemoLabel.FontWeight = 'bold';
            app.DemoLabel.Text = '答辩快捷样例';

            app.DemoDropDown = uidropdown(app.ControlPanel);
            app.DemoDropDown.ValueChangedFcn = createCallbackFcn(app, @DemoDropDownValueChanged, true);
            app.DemoDropDown.Position = [24 595 232 26];
            app.DemoDropDown.Items = {'正常通过 - S01 走廊 004'};

            app.FilterLabel = uilabel(app.ControlPanel);
            app.FilterLabel.Position = [24 548 220 22];
            app.FilterLabel.FontWeight = 'bold';
            app.FilterLabel.Text = '全部正式样本';

            app.SampleFilterDropDown = uidropdown(app.ControlPanel);
            app.SampleFilterDropDown.Items = {'全部样本', '正常测试', '错误口令', '冒充用户', '边界样例'};
            app.SampleFilterDropDown.ValueChangedFcn = createCallbackFcn(app, @SampleFilterDropDownValueChanged, true);
            app.SampleFilterDropDown.Position = [24 518 232 26];
            app.SampleFilterDropDown.Value = '全部样本';

            app.AllSampleLabel = uilabel(app.ControlPanel);
            app.AllSampleLabel.Position = [24 484 220 22];
            app.AllSampleLabel.Text = '从 CSV 选择任意评估音频';

            app.AllSampleDropDown = uidropdown(app.ControlPanel);
            app.AllSampleDropDown.ValueChangedFcn = createCallbackFcn(app, @AllSampleDropDownValueChanged, true);
            app.AllSampleDropDown.Position = [24 454 232 26];
            app.AllSampleDropDown.Items = {'结果表未加载'};

            app.RunDemoBTN = uibutton(app.ControlPanel, 'push');
            app.RunDemoBTN.ButtonPushedFcn = createCallbackFcn(app, @RunDemoBTNButtonPushed, true);
            app.RunDemoBTN.Position = [34 397 210 38];
            app.RunDemoBTN.Text = '一键演示';
            app.RunDemoBTN.FontSize = 16;
            app.RunDemoBTN.FontWeight = 'bold';
            app.RunDemoBTN.BackgroundColor = '#00A087';
            app.RunDemoBTN.FontColor = '#FFFFFF';

            app.LoadBTN = uibutton(app.ControlPanel, 'push');
            app.LoadBTN.ButtonPushedFcn = createCallbackFcn(app, @LoadBTNButtonPushed, true);
            app.LoadBTN.Position = [34 345 98 28];
            app.LoadBTN.Text = '手动选音频';

            app.PlayBTN = uibutton(app.ControlPanel, 'push');
            app.PlayBTN.ButtonPushedFcn = createCallbackFcn(app, @PlayBTNButtonPushed, true);
            app.PlayBTN.Position = [146 345 98 28];
            app.PlayBTN.Text = '播放音频';

            app.ExtractBTN = uibutton(app.ControlPanel, 'push');
            app.ExtractBTN.ButtonPushedFcn = createCallbackFcn(app, @ExtractBTNButtonPushed, true);
            app.ExtractBTN.Position = [34 305 98 26];
            app.ExtractBTN.Text = '特征提取';

            app.MatchBTN = uibutton(app.ControlPanel, 'push');
            app.MatchBTN.ButtonPushedFcn = createCallbackFcn(app, @MatchBTNButtonPushed, true);
            app.MatchBTN.Position = [146 305 98 26];
            app.MatchBTN.Text = '运行识别';

            app.EnvLabel = uilabel(app.ControlPanel);
            app.EnvLabel.HorizontalAlignment = 'right';
            app.EnvLabel.Position = [39 245 60 22];
            app.EnvLabel.Text = '采集环境';

            app.EnvDropDown = uidropdown(app.ControlPanel);
            app.EnvDropDown.Items = {'未选择'};
            app.EnvDropDown.Position = [112 245 132 24];
            app.EnvDropDown.Value = '未选择';
            app.EnvDropDown.Enable = 'off';

            app.UserLabel = uilabel(app.ControlPanel);
            app.UserLabel.HorizontalAlignment = 'right';
            app.UserLabel.Position = [39 205 60 22];
            app.UserLabel.Text = '当前用户';

            app.TargetUserDropDown = uidropdown(app.ControlPanel);
            app.TargetUserDropDown.Items = {'未选择'};
            app.TargetUserDropDown.Position = [112 205 132 24];
            app.TargetUserDropDown.Value = '未选择';
            app.TargetUserDropDown.Enable = 'off';

            app.FileNameLabel = uilabel(app.ControlPanel);
            app.FileNameLabel.Position = [24 150 232 22];
            app.FileNameLabel.Text = '文件名:';

            app.SampleRateLabel = uilabel(app.ControlPanel);
            app.SampleRateLabel.Position = [24 116 232 22];
            app.SampleRateLabel.Text = '采样率:';

            app.SampleTypeLabel = uilabel(app.ControlPanel);
            app.SampleTypeLabel.Position = [24 82 232 22];
            app.SampleTypeLabel.Text = '样本类型: 未选择';

            app.HeaderPanel = uipanel(app.GridLayout);
            app.HeaderPanel.Title = '';
            app.HeaderPanel.Layout.Row = 1;
            app.HeaderPanel.Layout.Column = [1 3];

            app.TitleLabel = uilabel(app.HeaderPanel);
            app.TitleLabel.FontSize = 34;
            app.TitleLabel.FontWeight = 'bold';
            app.TitleLabel.FontColor = '#3C5488';
            app.TitleLabel.Position = [20 10 760 48];
            app.TitleLabel.Text = '轻量化声音锁-声纹认证原型系统';

            app.FeaturePanel = uipanel(app.GridLayout);
            app.FeaturePanel.Title = '声学特征展示';
            app.FeaturePanel.Layout.Row = 2;
            app.FeaturePanel.Layout.Column = 2;

            app.FeatureGrid = uigridlayout(app.FeaturePanel);
            app.FeatureGrid.ColumnWidth = {'1x'};
            app.FeatureGrid.RowHeight = {30, '0.75x', 38, '1x'};

            app.WaveformLabel = uilabel(app.FeatureGrid);
            app.WaveformLabel.FontSize = 18;
            app.WaveformLabel.FontWeight = 'bold';
            app.WaveformLabel.FontColor = '#3C5488';
            app.WaveformLabel.Layout.Row = 1;
            app.WaveformLabel.Layout.Column = 1;
            app.WaveformLabel.Text = '语音波形';

            app.WaveformAxes = uiaxes(app.FeatureGrid);
            xlabel(app.WaveformAxes, 'Time (s)')
            ylabel(app.WaveformAxes, 'Amplitude')
            app.WaveformAxes.Layout.Row = 2;
            app.WaveformAxes.Layout.Column = 1;

            app.MFCCLabel = uilabel(app.FeatureGrid);
            app.MFCCLabel.FontSize = 22;
            app.MFCCLabel.FontWeight = 'bold';
            app.MFCCLabel.FontColor = '#3C5488';
            app.MFCCLabel.Layout.Row = 3;
            app.MFCCLabel.Layout.Column = 1;
            app.MFCCLabel.Text = 'MFCC 特征';

            app.MFCCAxes = uiaxes(app.FeatureGrid);
            xlabel(app.MFCCAxes, 'Frame Index')
            ylabel(app.MFCCAxes, 'MFCC Coefficient Index')
            app.MFCCAxes.Layout.Row = 4;
            app.MFCCAxes.Layout.Column = 1;

            app.ResultPanel = uipanel(app.GridLayout);
            app.ResultPanel.Title = '识别结果';
            app.ResultPanel.Layout.Row = 2;
            app.ResultPanel.Layout.Column = 3;

            app.ResultTitleLabel = uilabel(app.ResultPanel);
            app.ResultTitleLabel.FontSize = 18;
            app.ResultTitleLabel.FontWeight = 'bold';
            app.ResultTitleLabel.FontColor = '#1F2937';
            app.ResultTitleLabel.Position = [25 590 180 24];
            app.ResultTitleLabel.Text = '最终判定';

            app.TextArea = uitextarea(app.ResultPanel);
            app.TextArea.Editable = 'off';
            app.TextArea.FontSize = 20;
            app.TextArea.FontWeight = 'bold';
            app.TextArea.Position = [25 460 340 120];
            app.TextArea.Value = {'请选择演示样例'};

            app.UITable = uitable(app.ResultPanel);
            app.UITable.ColumnName = {'文件名'; '类型'; '期望'; '声纹判定'; '口令验证'; '最终判定'; '是否正确'; '声纹距离'; '口令距离'};
            app.UITable.RowName = {};
            app.UITable.Position = [25 48 340 385];
            app.UITable.Data = {};

            app.UIFigure.Visible = 'on';
        end
    end

    methods (Access = public)

        function app = SRDPVoiceLockAppFixed
            createComponents(app)
            initializeAppState(app)
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        function delete(app)
            delete(app.UIFigure)
        end
    end
end
