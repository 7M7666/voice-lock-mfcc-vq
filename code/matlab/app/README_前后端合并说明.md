# MATLAB App 前后端合并说明

正式演示入口：

```text
code/matlab/app/run_voice_lock_app.m
code/matlab/app/SRDPVoiceLockAppFixed.m
```

旧版 `SRDPVoiceLockApp.mlapp` 和对应 `.m` 源码已归档到 `code/matlab/legacy_app/`，可以作为组员原始 App Designer 版本留档；不要用它验证本次演示台流程，因为正式入口是 `SRDPVoiceLockAppFixed.m`。

## 运行方式

1. 先在项目根目录运行后端流程：

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
- 前端不动态调用 Python，答辩演示采用固定后端结果，稳定性更高。
- 如果更换或新增音频，需要先重新运行后端 pipeline，确保 CSV 中存在对应 `file_path`。
