# MATLAB App 前后端合并说明

**结项说明（2026-11-01）**：本目录是中期 MATLAB/CSV 展示版本，不是当前正式 App。当前启动入口为项目根目录 `.\run.ps1 gui` 或 Python 直接运行 `code/python/gui.py`，详见[项目 README](../../../README.md)。以下保留历史架构和操作语境，不作为现场注册验证的证据。

历史演示入口：

```text
code/matlab/app/run_voice_lock_app.m
code/matlab/app/SRDPVoiceLockAppFixed.m
```

旧版 `SRDPVoiceLockApp.mlapp` 和对应 `.m` 源码已归档到 `code/matlab/legacy_app/`；`SRDPVoiceLockAppFixed.m` 是当时中期演示入口，两个 MATLAB 版本均留档。

## 运行方式

1. 历史数据生成命令如下，会覆盖默认模型、特征和部分结果，**本次结项不执行**；查看历史展示使用现有 CSV 即可：

```powershell
.\run.ps1 pipeline --metadata data/demo_audio/metadata_srdp.csv
```

2. 在 MATLAB 中打开并运行 `code/matlab/app/run_voice_lock_app.m`。
3. 答辩时在“答辩快捷样例”中选择样本，点击“一键演示”。
4. 如需换样本，可在“全部正式样本”中按类型筛选，也可以点击“手动选音频”选择 wav。
5. 前端会自动加载音频、画波形和 MFCC、读取 `results/tables/lbg_vq_recognition_results.csv`，并只显示当前样本的判定详情。

## 逻辑边界

- MATLAB 前端只做交互、播放、特征展示和结果表读取。
- Python 后端负责正式识别：LBG-VQ 声纹识别 + MFCC-DTW 口令验证。
- 前端不动态调用 Python，只读取固定结果；未列入结果表的 WAV 不能据此得到实时认证判定。
- 当前结项不更换本历史前端的音频和 CSV，不为它重跑 pipeline；现场录音注册、验证使用 Python App。
