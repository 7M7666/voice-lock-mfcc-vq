# MATLAB UI 制作说明

## 定位

中期答辩阶段 MATLAB 没必要再完整重写一套 MFCC+LBG-VQ 主算法。现在 Python 已经承担主实验流水线，MATLAB 建议做成可视化 GUI 和交叉验证展示，用来证明系统有原型界面。

## 最小交付

张新宇负责在 MATLAB App Designer 中做一个轻量界面，交付：

- `code/matlab/SRDPVoiceLockApp.mlapp`
- `results/figures/matlab_ui_main.png`
- `results/figures/matlab_ui_result.png`

如果时间来不及，至少做出界面截图和一张波形/MFCC 展示图。

## 界面布局

建议控件：

- `选择音频` 按钮
- `播放` 按钮
- `环境模式` 下拉框：宿舍 / 走廊 / 实验室
- `注册用户` 下拉框：S01 / S02 / S03
- 波形坐标轴
- MFCC 热力图坐标轴
- 识别结果文本框或状态灯
- 结果表格，显示最近一次 Python 输出的识别结果

## 最快实现路线

1. 点击 `选择音频` 后，用 `uigetfile` 读取 WAV。
2. 用 `audioread` 和 `plot` 显示波形。
3. 如果 MATLAB 安装了 Audio Toolbox，用 `mfcc` 函数显示 MFCC；如果没有，就先只显示波形和能量曲线。
4. 识别结果不必在 MATLAB 里重写算法，可以读取 Python 已生成的结果表：

```matlab
T = readtable("results/tables/lbg_vq_recognition_results.csv");
lastRow = T(end, :);
```

5. 在 UI 里显示：
   - `true_speaker`
   - `expected_decision`
   - `pred_speaker`
   - `decision`
   - `distance`
   - `threshold`

## 可选：MATLAB 调用 Python 流水线

如果想在 App 里放一个 `运行识别` 按钮，可以调用：

```matlab
cmd = 'powershell -ExecutionPolicy Bypass -File run.ps1 pipeline --metadata data/demo_audio/metadata_srdp.csv';
system(cmd);
```

注意：这个按钮只作为演示用，答辩时更稳的做法是提前跑好 Python 结果，MATLAB UI 只读取 CSV 并展示。

## 截图要求

PPT 里至少放 1 张 MATLAB UI 截图，截图里要能看到：

- 选中的音频文件名；
- 波形图；
- MFCC 或能量图；
- 环境模式；
- 识别结果或 `REJECT`。

