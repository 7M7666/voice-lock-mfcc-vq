# SRDP 轻量化声音锁项目工程

本目录是中期答辩用的正式工程目录，当前只保留正式 SRDP 自采数据、后端识别流水线、MATLAB 前端入口、正式结果和说明文档。

## 目录结构

- `code/python/`：后端主程序 `srdp.py`，包含 SNR 估计、特征提取、MFCC + LBG-VQ 声纹识别、MFCC-DTW 口令二次验证和图表生成。
- `code/matlab/app/`：MATLAB 前端正式演示入口，只保留 `run_voice_lock_app.m`、`SRDPVoiceLockAppFixed.m` 和联调说明。
- `code/matlab/legacy_app/`：组员原始 App Designer 文件和旧 `.m` 源码归档，不作为正式演示入口。
- `code/matlab/legacy_preprocess/`、`code/legacy_notebooks/`：早期 MATLAB 预处理脚本和实验 notebook 归档，正式识别以 Python 后端为准。
- `data/demo_audio/`：正式数据标注表 `metadata_srdp.csv` 和 99 条自采 wav，位于 `srdp_dataset/`。
- `data/features/`：后端生成的正式特征文件 `srdp_features.npz`。
- `models/`：正式 LBG-VQ 码本模型 `lbg_vq_codebooks.npz`。
- `results/`：正式识别结果表、指标表、混淆矩阵和可视化图。
- `docs/`：采集规范、答辩分工、PPT 和 MATLAB UI 说明等辅助材料；过期缺口记录放在 `docs/archive/`。

## 后端单条推理接口

结题 GUI 可以直接复用 Python 后端，不需要重新写一套判定逻辑。核心函数位于 `code/python/srdp.py`：

- `load_model()`：加载 `models/lbg_vq_codebooks.npz` 和 `data/features/srdp_features.npz`，恢复码本、阈值、DTW 训练模板和融合参数。
- `extract_one_audio_features(wav_path, model)`：对单条 WAV 提取特征、估计 SNR、计算质量分，并套用训练集 z-score。
- `predict_one(wav_path, model=None, metadata=None)`：输出 `decision`、`pred_speaker`、`voice_score`、`passphrase_score`、`fusion_score`、`reject_reason` 等字段。

命令行快速测试：

```powershell
.\run.ps1 predict-one data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav --expected-decision S01 --speaker-id S01 --split test
```

## 当前正式效果

当前正式配置是 `MFCC + LBG-VQ codebook=16`，并加入 `MFCC-DTW` 作为第二层口令相似度验证。

| 方案 | 样本范围 | 正确数 | 准确率 |
|---|---:|---:|---:|
| 单层 LBG-VQ 声纹识别 | 全部评估样本 | 52/63 | 82.5% |
| 两层验证：声纹 + 口令 | 全部评估样本 | 56/63 | 88.9% |
| 两层验证 | 注册用户正常测试 | 17/24 | 70.8% |
| 两层验证 | 错误口令拒识 | 36/36 | 100.0% |
| 两层验证 | 非注册用户冒充拒识 | 3/3 | 100.0% |

答辩建议表述：系统已经完成“自采录音 -> 特征提取 -> 声纹识别 -> 口令二次验证 -> 接受/拒绝”的闭环。当前结果足够支撑中期答辩，但不建议说成商用级成熟系统；更稳妥的说法是“初步有效，错误口令和冒充样本拒识表现较好，注册用户正常测试仍有误拒，后续继续优化阈值和扩充数据”。

## 一键运行后端

在项目根目录运行，下面两种写法等价，默认已经指向正式 `metadata_srdp.csv`：

```powershell
.\run.ps1 pipeline
.\run.ps1 pipeline --metadata data/demo_audio/metadata_srdp.csv
```

这条命令会重新生成：

```text
results/tables/snr_estimates.csv
data/features/srdp_features.npz
results/tables/feature_summary.csv
results/tables/lbg_vq_recognition_results.csv
results/tables/voice_lock_metrics.csv
results/figures/environment_feature_overview.png
results/figures/snr_by_environment.png
results/figures/accuracy_by_split.png
results/figures/lbg_vq_confusion_matrix.png
results/figures/lbg_vq_voice_only_confusion_matrix.png
results/figures/passphrase_dtw_distance.png
results/figures/error_breakdown.png
results/figures/two_layer_pipeline.png
models/lbg_vq_codebooks.npz
```

如果换电脑后没有 `code/python/venv/`，先在项目根目录执行：

```powershell
python -m venv code/python/venv
.\code\python\venv\Scripts\python.exe -m pip install -r code/python/requirements.txt
```

## 批量对比实验

结题阶段用于补充“多特征融合”和“码本大小”对比表时，可运行：

```powershell
.\run.ps1 experiments
```

默认会复用 `data/features/srdp_features.npz`，批量比较：

- 特征组合：`mfcc`、`mfcc_delta`、`all`
- Codebook size：`8`、`16`、`32`

每组实验的逐条结果、指标表和图会保存到 `results/experiments/`，汇总对比表保存到：

```text
results/tables/experiment_matrix.csv
```

如果新增或修改了 metadata，可加 `--refresh-features` 重新提取特征：

```powershell
.\run.ps1 experiments --refresh-features
```

## DTW 策略和融合校准

`features` 命令会在 `feature_summary.csv` 中记录 `estimated_snr_db`、`quality_score` 和 `low_quality`；`vq` 命令支持在不改默认基线的前提下切换口令分支特征、DTW 多模板聚合方式、约束窗口、绕路惩罚，并在逐条结果中输出 `voice_score`、`passphrase_score`、`quality_score` 和 `fusion_score`。

比较不同口令 DTW 策略：

```powershell
.\run.ps1 dtw-experiments
```

输出：

```text
results/tables/dtw_strategy_matrix.csv
results/dtw_experiments/
```

系统扫描融合权重、融合阈值和音频质量权重：

```powershell
.\run.ps1 fusion-experiments
```

输出：

```text
results/tables/fusion_matrix.csv
results/tables/recommended_config.csv
results/fusion_experiments/
```

当前安全优先筛选结果显示，`MFCC + codebook=16 + 全部 31 维特征 DTW` 可以在 False Accept 为 0 的前提下，把总体准确率提高到 90.5%，正常用户通过率提高到 75.0%，False Reject 降到 6 条。`quality_weight=0.2` 已纳入融合矩阵，当前数据下不会进一步改善最佳结果，说明现有误接受主要不是低质量录音造成的；后续新增远距离、低音量、背景噪声和回放样本后，这个质量分支会更有价值。`all + codebook=8` 仍然是高通过率候选，但会出现 1 条 False Accept，因此不能直接替代正式安全基线。

如果只需要根据已有融合矩阵重新生成推荐配置表：

```powershell
.\run.ps1 recommend-config
```

## 最小验证

结题前可运行 smoke test，检查 metadata、音频文件、pipeline 输出和安全优先推荐配置：

```powershell
.\run.ps1 smoke-test
```

如果只是检查已有输出，不重跑 pipeline：

```powershell
.\run.ps1 smoke-test --skip-pipeline
```

## 启动 MATLAB 前端

1. 先运行后端 pipeline，确保 `results/tables/lbg_vq_recognition_results.csv` 是最新的。
2. 在 MATLAB 中打开并运行：

```text
code/matlab/app/run_voice_lock_app.m
```

3. 答辩时优先在“答辩快捷样例”中选择样本，点击“一键演示”。
4. 如需临时换样本，可在“全部正式样本”中按类型筛选 63 个评估音频，或点击“手动选音频”选择 wav。
5. 前端会自动加载音频、画波形和 MFCC、读取后端 CSV，并只显示当前样本的最终判定和详情。

前端只负责演示、播放和特征展示；正式识别逻辑全部在 Python 后端。答辩演示建议采用快捷样例 + 固定后端结果，稳定性比现场动态调用 Python 更高。旧版 `SRDPVoiceLockApp.mlapp` 已归档到 `code/matlab/legacy_app/`，正式演示只运行 `SRDPVoiceLockAppFixed.m`。

## 启动 Python GUI

结题阶段推荐使用 PySide6 一体化界面进行现场展示：

```powershell
.\run.ps1 gui
```

GUI 默认复用 Python 后端模型和特征文件，不重写识别逻辑：

- 模型：`models/lbg_vq_codebooks.npz`
- 特征：`data/features/srdp_features.npz`
- 单条推理入口：`predict_one()`

推荐演示顺序：

1. 在“答辩演示台”选择固定样例，依次展示正常通过、错误口令拒绝、非注册用户拒绝。
2. 点击“选择 WAV”可加载已有音频，点击“开始验证”运行识别。
3. “样本分组”和“测试场景”用于给结果加实验标签，便于解释正常测试、错误口令、非注册冒充、回放风险等场景，不会改变后端算法。
4. 打开“实验结果”，可直接查看当前正式指标和安全优先推荐配置。
5. 如老师希望现场注册，打开“临时注册”，先录入 3-5 条注册语音，点击“建立临时模型”，再录一条新语音验证这个临时用户。

临时注册只在当前 GUI 会话中生效，不会自动写入正式 metadata、模型或实验结果。

如果只想检查 GUI 后端能否加载模型并跑通固定样例：

```powershell
.\run.ps1 gui --smoke
```

## 正式演示推荐音频

前端已经内置以下快捷样例，答辩时直接从下拉框选择即可：

- 正常通过：`data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav`
- 正常通过：`data/demo_audio/srdp_dataset/S02/corridor/S02_corridor_005.wav`
- 错误口令拒绝：`data/demo_audio/srdp_dataset/S01_wrong/dorm/S01_wrong_003.wav`
- 非注册用户拒绝：`data/demo_audio/srdp_dataset/U01/dorm/U01_corridor_001.wav`
- 边界样例：`data/demo_audio/srdp_dataset/S01/lab/S01_lab_004.wav`

若更换演示音频，优先从“全部正式样本”列表选择；如果是新增 wav，先重新运行一次 pipeline，确保它出现在 `lbg_vq_recognition_results.csv` 中。

## 项目清理边界

旧版 MATLAB App、早期 MATLAB 预处理脚本、实验 notebook 和过期缺口记录已经归档到 `legacy` / `archive` 目录，避免与正式入口混淆，同时保留前期工作证据。

`code/python/venv/` 是本机运行环境，不建议提交到 Git；仓库已通过 `.gitignore` 忽略它。
