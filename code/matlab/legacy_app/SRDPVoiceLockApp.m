classdef SRDPVoiceLockApp < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure            matlab.ui.Figure
        GridLayout          matlab.ui.container.GridLayout
        Panel4              matlab.ui.container.Panel
        TextArea            matlab.ui.control.TextArea
        TextAreaLabel       matlab.ui.control.Label
        UITable             matlab.ui.control.Table
        Panel3              matlab.ui.container.Panel
        GridLayout2         matlab.ui.container.GridLayout
        MFCCLabel           matlab.ui.control.Label
        Label_6             matlab.ui.control.Label
        MFCCAxes            matlab.ui.control.UIAxes
        WaveformAxes        matlab.ui.control.UIAxes
        Panel2              matlab.ui.container.Panel
        Label               matlab.ui.control.Label
        Panel               matlab.ui.container.Panel
        SampleRateLabel     matlab.ui.control.Label
        FileNameLabel       matlab.ui.control.Label
        TargetUserDropDown  matlab.ui.control.DropDown
        Label_3             matlab.ui.control.Label
        EnvDropDown         matlab.ui.control.DropDown
        Label_2             matlab.ui.control.Label
        MatchBTN            matlab.ui.control.Button
        ExtractBTN          matlab.ui.control.Button
        PlayBTN             matlab.ui.control.Button
        LoadBTN             matlab.ui.control.Button
    end

    
    properties (Access = private)
        AudioPath            % 音频文件路径
        AudioData            % 音频数据矩阵
        SampleRate           % 采样率
        FeaturesExtracted = false % 防呆锁，记录是否已提取特征
    end
    

    % Callbacks that handle component events
    methods (Access = private)

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

        % Button pushed function: LoadBTN
        function LoadBTNButtonPushed(app, event)
            [file, path] = uigetfile({'*.wav', 'WAV Audio (*.wav)'});
            if isequal(file, 0)
                return;
            end

            app.AudioPath = fullfile(path, file);
            [y, fs] = audioread(app.AudioPath);
            if size(y, 2) > 1
                y = mean(y, 2);
            end
            app.AudioData = y;
            app.SampleRate = fs;

            app.FileNameLabel.Text = ['文件名: ', file];
            app.SampleRateLabel.Text = ['采样率: ', num2str(fs), ' Hz'];

            plot(app.WaveformAxes, (0:length(y)-1) / fs, y, 'Color', '#3C5488', 'LineWidth', 0.8);
            app.WaveformAxes.XLim = [0, length(y)/fs];
            grid(app.WaveformAxes, 'on');

            app.FeaturesExtracted = false; 
            app.TextArea.Value = {
                '已导入音频'
                ['文件名: ', file]
                '下一步: 点击“特征提取”'
            };
            app.TextArea.BackgroundColor = '#FFFFFF';
            app.TextArea.FontColor = '#1F2937';
            cla(app.MFCCAxes);
            app.UITable.Data = table();
        end

        % Button pushed function: PlayBTN
        function PlayBTNButtonPushed(app, event)
            if isempty(app.AudioData)
                uialert(app.UIFigure, '请先导入语音文件！', '提示');
                return;
            end
            % 使用 audioplayer 播放，防止界面卡死
            player = audioplayer(app.AudioData, app.SampleRate);
            play(player);
            assignin('base', 'temp_player', player);
        end

        % Button pushed function: ExtractBTN
        function ExtractBTNButtonPushed(app, event)
            if isempty(app.AudioData)
                uialert(app.UIFigure, '请先点击【选择音频】导入语音文件！', '操作提示', 'Icon', 'warning');
                return;
            end
            
            app.UIFigure.Pointer = 'watch';
            drawnow;
            
            try
                % 前端只做特征展示，正式识别以后端 Python 结果为准。
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
                
                app.FeaturesExtracted = true;
                app.TextArea.Value = {
                    '声学特征已提取'
                    '前端展示: 波形 + MFCC'
                    '正式识别: 读取 Python 后端 CSV'
                };
                app.TextArea.FontColor = '#1F2937';
                
            catch ME
                uialert(app.UIFigure, ['特征提取失败: ', ME.message], '系统错误', 'Icon', 'error');
                app.FeaturesExtracted = false; 
            end
            
            app.UIFigure.Pointer = 'arrow';
        end

        % Button pushed function: MatchBTN
        function MatchBTNButtonPushed(app, event)
            if ~app.FeaturesExtracted
                uialert(app.UIFigure, '请先点击【提取特征】！不能跳步哦。', '操作顺序错误', 'Icon', 'warning');
                return; 
            end
            
            d = uiprogressdlg(app.UIFigure, 'Title', '执行声纹匹配', ...
                'Message', '正在读取后端两层声音锁结果...', 'Indeterminate', 'on');

            appFile = mfilename('fullpath');
            if strlength(string(appFile)) > 0
                appDir = fileparts(appFile);
            else
                appDir = pwd;
            end
            projectCandidates = {};
            projectCandidates = app.appendRootAndParents(projectCandidates, pwd);
            projectCandidates = app.appendRootAndParents(projectCandidates, appDir);
            if ~isempty(app.AudioPath)
                projectCandidates = app.appendRootAndParents(projectCandidates, fileparts(app.AudioPath));
            end

            candidates = {};
            seenRoots = {};
            for rootIdx = 1:numel(projectCandidates)
                root = projectCandidates{rootIdx};
                if strlength(string(root)) == 0
                    continue;
                end
                if any(strcmp(seenRoots, root))
                    continue;
                end
                seenRoots{end + 1, 1} = root; %#ok<AGROW>
                candidates{end + 1, 1} = fullfile(root, 'results', 'tables', 'lbg_vq_recognition_results.csv'); %#ok<AGROW>
            end

            resultPath = "";
            for i = 1:numel(candidates)
                if isfile(candidates{i})
                    resultPath = string(candidates{i});
                    break;
                end
            end

            if resultPath == ""
                close(d);
                expectedRel = fullfile('results', 'tables', 'lbg_vq_recognition_results.csv');
                app.TextArea.Value = {
                    '尚未找到 Python 结果表'
                    ['期望文件: ', expectedRel]
                    '请先在项目根目录运行: .\run.ps1 pipeline'
                };
                uialert(app.UIFigure, sprintf('找不到 %s。\n请先在项目根目录运行 .\\run.ps1 pipeline。', expectedRel), '文件缺失', 'Icon', 'error');
                return;
            end

            try
                T = readtable(resultPath, 'TextType', 'string');
                [~, audioName, audioExt] = fileparts(app.AudioPath);
                audioFile = string(audioName) + string(audioExt);
                filePaths = string(T.file_path);
                normalizedAudioPath = replace(string(app.AudioPath), '\', '/');
                idx = find(endsWith(normalizedAudioPath, filePaths), 1, 'first');
                if isempty(idx)
                    idx = find(endsWith(filePaths, audioFile), 1, 'first');
                end
                if isempty(idx)
                    close(d);
                    app.TextArea.Value = {
                        '当前音频不在正式评估结果表'
                        ['音频: ', char(audioFile)]
                        '若这是训练样本，请选择 *_004.wav、*_005.wav、wrong 或 U01 测试样本'
                    };
                    warningMessage = sprintf('CSV 中找不到当前音频: %s\n该音频可能是训练样本，请选择 *_004.wav、*_005.wav、wrong 或 U01 测试样本。', char(audioFile));
                    uialert(app.UIFigure, warningMessage, '未匹配到结果', 'Icon', 'warning');
                    return;
                end

                row = T(idx, :);
                displayCols = ["file_path", "expected_decision", "voice_decision", ...
                    "passphrase_passed", "decision", "correct", "split", "environment"];
                displayCols = displayCols(ismember(displayCols, string(T.Properties.VariableNames)));
                app.UITable.Data = T(:, displayCols);
                app.UITable.ColumnName = {'音频路径'; '期望'; '声纹判定'; '口令通过'; '最终判定'; '正确'; '类型'; '环境'};

                passText = char(string(row.passphrase_passed));
                if isempty(passText)
                    passText = '声纹层已拒绝';
                end
                decisionText = char(string(row.decision));
                voiceDecisionText = char(string(row.voice_decision));
                expectedText = char(string(row.expected_decision));
                distanceText = char(string(row.distance));
                thresholdText = char(string(row.threshold));
                passDistanceText = char(string(row.passphrase_distance));
                passThresholdText = char(string(row.passphrase_threshold));

                isCorrect = strcmpi(char(string(row.correct)), 'true');
                resultMark = '通过';
                if strcmp(decisionText, 'REJECT')
                    resultMark = '拒绝';
                end

                app.TextArea.Value = {
                    ['音频: ', char(audioFile)]
                    ['最终判定: ', decisionText, '（', resultMark, '）']
                    ['声纹判定: ', voiceDecisionText]
                    ['口令验证: ', passText]
                    ['期望结果: ', expectedText]
                    ['声纹距离: ', distanceText, ' / ', thresholdText]
                    ['口令距离: ', passDistanceText, ' / ', passThresholdText]
                };

                app.TextArea.FontSize = 14;
                app.TextArea.FontWeight = 'bold';
                if isCorrect
                    app.TextArea.FontColor = '#16803C';
                    app.TextArea.BackgroundColor = '#F0FFF4';
                else
                    app.TextArea.FontColor = '#C0392B';
                    app.TextArea.BackgroundColor = '#FFF5F5';
                end
            catch ME
                uialert(app.UIFigure, ['读取或显示识别结果失败: ', ME.message], '错误', 'Icon', 'error');
            end

            if isvalid(d)
                close(d);
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Color = [0.9569 0.9686 0.9765];
            app.UIFigure.Position = [100 100 1280 800];
            app.UIFigure.Name = 'SRDP 声音锁演示系统';

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {260, '1x', 380};
            app.GridLayout.RowHeight = {80, '1x'};

            % Create Panel
            app.Panel = uipanel(app.GridLayout);
            app.Panel.Title = '控制台';
            app.Panel.Layout.Row = 2;
            app.Panel.Layout.Column = 1;

            % Create LoadBTN
            app.LoadBTN = uibutton(app.Panel, 'push');
            app.LoadBTN.ButtonPushedFcn = createCallbackFcn(app, @LoadBTNButtonPushed, true);
            app.LoadBTN.Position = [80 613 100 23];
            app.LoadBTN.Text = '选择音频';

            % Create PlayBTN
            app.PlayBTN = uibutton(app.Panel, 'push');
            app.PlayBTN.ButtonPushedFcn = createCallbackFcn(app, @PlayBTNButtonPushed, true);
            app.PlayBTN.Position = [80 550 100 23];
            app.PlayBTN.Text = '播放音频';

            % Create ExtractBTN
            app.ExtractBTN = uibutton(app.Panel, 'push');
            app.ExtractBTN.ButtonPushedFcn = createCallbackFcn(app, @ExtractBTNButtonPushed, true);
            app.ExtractBTN.Position = [80 485 100 23];
            app.ExtractBTN.Text = '特征提取';

            % Create MatchBTN
            app.MatchBTN = uibutton(app.Panel, 'push');
            app.MatchBTN.ButtonPushedFcn = createCallbackFcn(app, @MatchBTNButtonPushed, true);
            app.MatchBTN.Position = [80 420 100 23];
            app.MatchBTN.Text = '运行识别';

            % Create Label_2
            app.Label_2 = uilabel(app.Panel);
            app.Label_2.HorizontalAlignment = 'right';
            app.Label_2.Position = [52 309 53 22];
            app.Label_2.Text = '采集环境';

            % Create EnvDropDown
            app.EnvDropDown = uidropdown(app.Panel);
            app.EnvDropDown.Items = {'纯净', '宿舍', '走廊', '实验室'};
            app.EnvDropDown.Position = [120 309 100 22];
            app.EnvDropDown.Value = '纯净';

            % Create Label_3
            app.Label_3 = uilabel(app.Panel);
            app.Label_3.HorizontalAlignment = 'right';
            app.Label_3.Position = [52 249 53 22];
            app.Label_3.Text = '当前用户';

            % Create TargetUserDropDown
            app.TargetUserDropDown = uidropdown(app.Panel);
            app.TargetUserDropDown.Items = {'S01', 'S02', 'S03', 'S04'};
            app.TargetUserDropDown.Position = [120 249 100 22];
            app.TargetUserDropDown.Value = 'S01';

            % Create FileNameLabel
            app.FileNameLabel = uilabel(app.Panel);
            app.FileNameLabel.Position = [32 177 213 22];
            app.FileNameLabel.Text = '文件名:';

            % Create SampleRateLabel
            app.SampleRateLabel = uilabel(app.Panel);
            app.SampleRateLabel.Position = [32 129 213 22];
            app.SampleRateLabel.Text = '采样率:';

            % Create Panel2
            app.Panel2 = uipanel(app.GridLayout);
            app.Panel2.Title = '';
            app.Panel2.Layout.Row = 1;
            app.Panel2.Layout.Column = [1 3];

            % Create Label
            app.Label = uilabel(app.Panel2);
            app.Label.FontSize = 36;
            app.Label.FontWeight = 'bold';
            app.Label.FontColor = [0 0.4471 0.7412];
            app.Label.Position = [13 8 544 48];
            app.Label.Text = '轻量化声音锁-声纹认证原型系统';

            % Create Panel3
            app.Panel3 = uipanel(app.GridLayout);
            app.Panel3.Title = '声学特征展示';
            app.Panel3.Layout.Row = 2;
            app.Panel3.Layout.Column = 2;

            % Create GridLayout2
            app.GridLayout2 = uigridlayout(app.Panel3);
            app.GridLayout2.ColumnWidth = {'1x'};
            app.GridLayout2.RowHeight = {30, '0.75x', 40, '1x'};

            % Create WaveformAxes
            app.WaveformAxes = uiaxes(app.GridLayout2);
            xlabel(app.WaveformAxes, 'Time (s)')
            ylabel(app.WaveformAxes, 'Amplitude')
            zlabel(app.WaveformAxes, 'Z')
            app.WaveformAxes.Layout.Row = 2;
            app.WaveformAxes.Layout.Column = 1;

            % Create MFCCAxes
            app.MFCCAxes = uiaxes(app.GridLayout2);
            xlabel(app.MFCCAxes, 'Frame Index')
            ylabel(app.MFCCAxes, 'MFCC Coefficient Index')
            zlabel(app.MFCCAxes, 'Z')
            app.MFCCAxes.Layout.Row = 4;
            app.MFCCAxes.Layout.Column = 1;

            % Create Label_6
            app.Label_6 = uilabel(app.GridLayout2);
            app.Label_6.FontSize = 18;
            app.Label_6.FontWeight = 'bold';
            app.Label_6.FontColor = [0.0745 0.6235 1];
            app.Label_6.Layout.Row = 1;
            app.Label_6.Layout.Column = 1;
            app.Label_6.Text = '语音波形';

            % Create MFCCLabel
            app.MFCCLabel = uilabel(app.GridLayout2);
            app.MFCCLabel.FontSize = 24;
            app.MFCCLabel.FontWeight = 'bold';
            app.MFCCLabel.FontColor = [0.0745 0.6235 1];
            app.MFCCLabel.Layout.Row = 3;
            app.MFCCLabel.Layout.Column = 1;
            app.MFCCLabel.Text = 'MFCC特征';

            % Create Panel4
            app.Panel4 = uipanel(app.GridLayout);
            app.Panel4.Title = '识别结果';
            app.Panel4.Layout.Row = 2;
            app.Panel4.Layout.Column = 3;

            % Create UITable
            app.UITable = uitable(app.Panel4);
            app.UITable.ColumnName = {'音频路径'; '期望'; '声纹判定'; '口令通过'; '最终判定'; '正确'; '类型'; '环境'};
            app.UITable.RowName = {};
            app.UITable.Position = [27 41 325 293];

            % Create TextAreaLabel
            app.TextAreaLabel = uilabel(app.Panel4);
            app.TextAreaLabel.HorizontalAlignment = 'right';
            app.TextAreaLabel.FontSize = 18;
            app.TextAreaLabel.Position = [39 537 96 24];
            app.TextAreaLabel.Text = '最终判定';

            % Create TextArea
            app.TextArea = uitextarea(app.Panel4);
            app.TextArea.FontSize = 18;
            app.TextArea.Position = [150 503 150 60];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = SRDPVoiceLockApp

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end
