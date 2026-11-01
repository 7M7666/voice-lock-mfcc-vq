# Voice Lock · MFCC + VQ

基于 **MFCC、LBG-VQ 与 DTW** 的轻量化声音锁原型，支持本地用户注册、麦克风录音验证、WAV 文件推理和离线对比实验。项目来自 SRDP，当前主要入口为 Python + PySide6 桌面应用，保留早期 MATLAB 演示与实验材料。

## 核心功能

- **声纹与口令双重验证**：LBG-VQ 比较说话人声纹，DTW 比较口令模板的声学相似度，两路共同参与接受/拒绝判定。
- **本地注册与验证**：录入至少三条合格语音建立用户模型，支持注册质量提示、验证记录、录音复核和用户删除。
- **可解释的结果展示**：查看声纹/口令分数、波形、MFCC、SNR 估计和拒绝原因。
- **实验与复现**：随工程提供 99 条自采 WAV、标注、基线模型和结果，支持特征组合、码本大小、DTW、阈值校准与融合对比。

```text
WAV / 麦克风录音 → 语音质量检查与预处理 → 特征提取与归一化
                                      ├─ MFCC + LBG-VQ：声纹匹配
                                      └─ 特征序列 + DTW：口令模板匹配
                                                 ↓
                                          接受 / 拒绝及分数
```

口令分支比较声学模板，**不将语音识别成文字**。当前项目用于研究和教学演示，尚未完成跨设备、相近基频陌生人及回放攻击等独立验收。

## 快速开始

以下命令在 Windows PowerShell 中运行，本机已验证的 Python 版本为 **3.11.9**。

```powershell
git clone https://github.com/7M7666/voice-lock-mfcc-vq.git
cd voice-lock-mfcc-vq
python -m venv code/python/venv
.\code\python\venv\Scripts\python.exe -m pip install -r code/python/requirements.txt
.\run.ps1 gui
```

启动后点击“注册新用户”，逐次录入至少三条合格的“芝麻开门”，完成注册后回到“声音验证”录音验证。“项目实验”可使用已有 WAV 演示并查看离线结果。现场录音需要可用的麦克风及系统录音权限；已有音频演示可直接使用仓库样本。

无需打开界面的单条推理与基本检查：

```powershell
.\run.ps1 predict-one data/demo_audio/srdp_dataset/S01/corridor/S01_corridor_004.wav --expected-decision S01 --speaker-id S01 --split test
.\run.ps1 gui --smoke
.\run.ps1 smoke-test --skip-pipeline
```

## 工程目录

| 路径 | 内容 |
|---|---|
| `code/python/` | 识别后端、PySide6 应用、实验脚本和自动测试 |
| `data/demo_audio/` | 99 条正式自采录音及 `metadata_srdp.csv` 标注 |
| `data/features/`、`models/` | 已保存的特征、归一化参数和码本模型 |
| `results/` | 指标表、逐条判定、图表和离线对照实验 |
| `docs/` | 采集规范、开发主线和项目分工 |
| `code/matlab/`、`code/legacy_notebooks/` | 历史前端、预处理与 notebook 归档 |
| `run.ps1` | Windows 命令行与 GUI 启动入口 |

## 结果与当前进度

正式数据包括 **36 条训练录音和 63 条评估录音**。仓库保留的历史双层验证基线为：总体正确 **56/63（88.9%）**，正常用户通过 **17/24**，错误口令拒绝 **36/36**，非注册用户拒绝 **3/3**，详见 [原始指标表](results/tables/voice_lock_metrics.csv)。这些数字来自小规模离线数据，不能代表当前现场注册用户的表现；评估集已多次用于开发，也不能视为新的独立泛化测试。

截至 **2026-11-01**，App 仍使用原双门槛判定；最新融合候选已经冻结，尚未部署，仍待第二轮独立录音与真实设备验收。2026-10-29 修复了 LBG 训练实现，重新执行 `pipeline` 会覆盖特征、默认模型和结果，不应预期复现修复前的历史数值。后续实验的证据边界和复现命令见下方详细说明及 [结题改造主线](docs/结题改造主线.md)。

## 本地数据与验证

本仓库保留正式实验数据，`app_data/` 中的现场用户档案、验证历史，`results/gui_history/` 中的导出记录，以及 `tmp*/` 临时录音与诊断材料不提交。详细开发记录中引用这些本地路径的实验，需要自行准备对应输入。

运行自动回归：

```powershell
.\code\python\venv\Scripts\python.exe -m unittest discover -s code/python -p 'test_*.py' -v
```

依赖列表见 [requirements.txt](code/python/requirements.txt)，当前未锁定各依赖版本。

---

## 开发记录与详细使用说明

本目录包含 SRDP 自采数据、Python 识别流水线与一体化 App、实验结果和说明文档；MATLAB 前端保留为历史演示。

2026-11-01最新开发筛选：在冻结的lwl2/lyh2模型和24条校准WAV上，新增相对身份距离、局部DTW及基频对照。已冻结一个**判定级融合候选**：原声纹≤1、原口令≤1.1、原Delta口令≤1、相对基频距离≤1。拟合及分组留一均为lwl2本人4/6、lyh2本人6/6，三类负例各0/12；相对原方案救回两人第9条、拦住3次原误接受，没有新增本人拒绝。单独Delta仍误接受5次其他人正确口令，单独相对基频仍误接受2次本人错误口令，组合在本批开发数据上互补；它与相同基频门槛后的最佳单流Delta完全相同，局部特征加权没有额外收益。候选仅供下一轮独立验收，**未部署，未证明泛化或身份/文字内容可靠分离**。两人的基频差异较大，尚无相近基频陌生人证据。

最终开发证据及冻结参数：`tmp/registration_diagnosis/20261101/both_speakers_structure_frozen/`，其中`frozen_candidate.json`绑定候选、对照及原独立验收协议，`freeze_receipt.json`记录哈希。实际App仍使用原双门槛，两份用户档案、历史、录音及正式99条数据保持不变。71项自动回归、GUI冒烟、3168次冻结参数重放通过；真实设备与第二轮24条新WAV尚未验收。以下段落保留各历史阶段语境，补录前的“下一步”不代表当前仍缺校准录音。

2026-11-01双人完整校准：已收到lwl2新12句，与确认顺序的lyh2十二句共同形成24条WAV、48次账户声明。原双门槛本人正确接受8/12、误接受3/36；多特征融合在拟合中为6/12、0/36，按WAV分组留一为5/12、1/36，最佳单流留一为6/12、1/36。融合尚未证明安全且不增加误拒的收益，不部署、不覆盖实际档案。旧lwl2漏录11句仍不参与拟合；第二轮独立验收尚未采集。详细逐条变化及分数重叠分析见主线文档最新记录。

2026-11-01后续清理：按用户明确要求，已删除旧lwl/lyh模型、原临时录音及个人诊断副本，仅保留lwl2/lyh2的现场档案和原有32条关联录音。正式99条数据与正式实验结果不变；历史文档中的旧现场结果保留为文字记录，但相应个人录音和模型不再能重放。当前两份模型保持冻结，下一步补lwl2完整12句校准，再以两人另录的独立样本做配对验收，见主线文档的最新实验协议。

2026-11-01进展：只分析新账户lwl2、lyh2，已从导出记录恢复lyh2的12条和lwl2的11条校准录音。用户确认lyh2顺序无误；lwl2漏录位置未知，内容不参与拟合。新增现场多特征融合及按WAV分组留一对照，校准拟合有收益但留一未证明改善，未替换实际档案。删除用户现可勾选同时清理App中明确关联的录音、历史、导出记录、注册报告、快照和备份，确认前显示数量；默认仍只删注册模型。共用文件、无归属录音及独立实验副本保留。详见主线文档最新记录。

2026-10-31阶段现场状态：18:29后lwl、lyh重新注册，各4条注册录音；用户确认新8次验证结果正确、全部说“芝麻开门”，据此记本人4次接受、其他人4次拒绝。旧档案2/5、2/7不是本轮新档案效果。尚无这两人的本人错误口令验收，不能宣称误拒已全部解决。

口令提示现称“口令模板匹配”，比较声学相似度，没有识别成文字；失败不能直接解释为说错内容。个人/共享模板与仅口令CMN的1040次开发声明对照发现，共享方案虽改善跨说话人正确口令匹配，却新增身份误接受，均未部署；实际档案和阈值保留。最新证据、逐条结果与复现入口见 [结题改造主线](docs/结题改造主线.md#2026-10-31-重新注册核对与口令内容混杂对照)。

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

## 历史正式数据基线（非现场新档案）

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

## 新一轮 VQ 与阈值校准实验

2026-10-29 已修复 LBG 细化在首轮提前退出的问题。已有88.9%/90.5%结果和默认模型作为历史基线保留，没有用新实验覆盖；重新执行 `pipeline` 会采用修复后的实现，不能预设仍复现旧数值。

```powershell
.\run.ps1 calibration-experiments
```

输出在 `results/calibration_experiments/`：`comparison.csv` 为7组对照，`calibration_trials.csv` 为训练内留一录音产生的36个本人、108个其他用户冒充声明，`thresholds.csv` 记录各用户阈值，`policy.json` 记录选参依据与候选选择；每组另有模型、逐条判定和指标表。

校准只使用36条训练录音，每折重新估计归一化并留出整条录音；最终63条评估不参与阈值/融合参数拟合。本轮比较修复训练、留一声纹阈值、声纹/口令均校准、绝对距离融合、保守口令融合、相对背景距离双门槛和相对距离融合。历史上该评估集已参与过选型，因此本轮是开发对照，不是全新的独立泛化测试。

本轮正常通过率最高且误拒较少的声纹校准候选达到总体93.7%、正常通过87.5%（21/24）、误拒3条，但误接受1条错误口令。相对距离方案没有在这批评估上额外改变判定；其融合权重0.5、阈值1.0亦无新增收益，所以训练选择明确标为 `cohort_two_layer`，不包装成融合提升。绝对距离融合虽然误接受为0，但误拒明显增加。所有新候选保持实验状态，没有自动替换默认模型。

App“项目实验 → 算法对比结果”可查看这7组结果。若要手工复核实验候选，可在“设置”把模型文件改为 `results/calibration_experiments/cohort_two_layer/model.npz`，点击“重新加载模型”，再使用已有音频演示或注册新的本地用户。此候选有已知误接受，不作为正式安全配置；已保存的用户继续使用自己的模型，需要重新注册才应用新设置。

新注册的声纹阈值已改用整条录音留一估计，不再使用原先训练距离乘1.5。加载校准候选时，注册模型会保存相对距离背景码本及判定参数。批量和App验证共用分数判定函数；新实验模型包含归一化、质量参数和口令模板，可以独立加载，避免依赖变动的特征文件。

```powershell
.\code\python\venv\Scripts\python.exe -m unittest discover -s code/python -p 'test_*.py' -v
.\run.ps1 gui --smoke
```

自动检查使用已有音频，不替代真实麦克风、相近错误口令和陌生人现场验收。当前训练录音均为正确口令，融合校准的负例来自其他注册用户，并不覆盖本人说错误口令；没有用36条错误口令评估样本反向调参。

## 后续路径、VAD、融合与归一化对照（2026-10-29）

```powershell
.\run.ps1 followup-experiments
```

结果独立保存在 `results/followup_experiments/`，不覆盖默认模型或原99条数据。可用 `--stage audit|dtw|preprocessing|fusion` 单独运行某阶段；分阶段复跑应指定独立 `--output-dir`，避免覆盖之前的总表。

- `nearest_paths.csv` / `alignment_pairs.csv` / `alignment_paths.png`：99条录音的最近模板及全部模板对齐路径分析。
- `comparison.csv`：12行统一校准、多模板/首条单模板/代表模板、连续步数限制、口令动态特征、帧级VAD、CMN/CMVN和声纹多特征对照。
- `fusion_comparison.csv` / `fusion_policies.csv` / `fusion_changes.csv`：双门槛、保留双门槛的融合、声纹有限互补、双方有限互补，及逐条新增拒绝/通过原因所需的分数。
- `audio_stress_summary.csv` / `audio_stress_trials.csv`：24条正常语音在原音量、0.1倍和0.01倍音量下，以及静音、直流、白噪声、有色噪声、纯音的合成压力测试；区分前置无语音拦截、最终拒绝、误拒、误接受和错认。
- `speaker_separation.csv`：归一化后本人/其他用户声纹分数分布与过门槛数量；每组另有模型、校准声明、逐条结果和分用户/环境指标。
- `protocol.json`：预设参数、训练/开发评估边界、计时范围及局限。

本轮只使用36条训练录音进行整条留一校准，每折重新拟合z-score；声纹阈值为均值+2倍标准差，口令为均值+1倍标准差。与历史表比较时要同时注明训练修复与校准方式；63条评估已反复用于开发，不能视为独立测试。DTW继续采用“欧氏距离之和/所选路径长度”，未改用tslearn默认的平方距离定义。新增Numba加速与路径回溯，已检查旧候选63条判定一致，分数误差不超过2e-6；计时排除首次JIT编译。

基准正常通过21/24、误接受1；连续步数≤5为22/24、误接受1，不能仅凭总体95.2%宣称安全提升。代表模板为23/24但误接受3；帧级VAD为20/24、误接受2；CMN为22/24、误接受3；CMVN为21/24、误接受1且误接受对象变为陌生人。三种训练内选参融合均为13/24、误接受0，相对双门槛额外拒绝8条正常语音并拦住1条错口令，没有证明互补收益。所有候选保持实验状态；App“项目实验→算法对比结果”显示这些开发对照。

本轮帧级VAD是能量/频谱平坦度筛选，保留内部停顿，只裁去外部非语音段；不是WebRTC VAD。它能拦截本轮白噪声，但有色噪声、纯音等仍可能经过前置筛选，需要后端拒绝，不能声称通用纯噪声检测。静音、直流及过短语音在WAV和现场用户验证入口明确返回拒绝。CMN/CMVN仅有同一数据集的消融，没有新设备泛化证据。各候选模型可在设置中加载后用WAV复核；代表/首条模板是该预训练模型保存的固定模板，现场重新注册会重新建立本人的多模板。

### 本人注册后全部拒绝的排查与修复

现场记录确认5次均为对应注册人说正确口令。两个已保存模型把口令校准阈值又乘以固定0.72，导致两个用户各自的3条注册录音留一验证也全部被口令分支拒绝。新注册现统一使用校准阈值本身（归一化通过线1.0），不再追加0.72。两个现有用户档案已备份到 `app_data/users/backups/` 后修正；原验证历史保留，不回写成通过。

重放这5条真实录音，修复前0/5通过，修复后2/5通过。两个用户各测试63条其他人录音，修复前后均拒绝，共126次。**仍有3条本人正确口令被拒绝，现场稳定性尚未解决**：其裁剪语音段比最近注册模板短，DTW有16–24步连续横/纵对齐；这支持继续排查语速/边界，但不能证明录音被截断，也不能据此继续放宽阈值。局部VAD/CMN候选未一致解决问题，没有替换现场用户参数。缺少这两人的独立错误口令录音，126条其他人对照不能证明本人错误口令安全性。排查明细保存在被Git忽略的 `tmp/registration_diagnosis/`，不混入正式实验。

已打开的App需要重新选择用户或重启，才能加载修正后的档案；新代码与新实验表需重启后使用。

2026-10-31 已开始现有录音上的软件改进，八项开发目标及进度见 `docs/结题改造主线.md` 顶部。新现场记录增加录音路径、原始距离/阈值、裁剪边界和模型参数指纹，并将对应用户模型快照保存到 `app_data/diagnostics/models/`。“验证记录→复核所选录音”可用当时模型与当前算法重新分析，不改写历史；导出保留诊断字段，旧记录不补造缺失信息。新代码需重启App生效。

本轮另发现：用录音末尾片段循环构造的10次低能量背景对照，原算法会全部接受。新增裁剪前能量起伏检查后全部拒绝，已检查原63条开发评估及5条本人录音的判定不变；本人误拒仍为3/5。该检查是开发阶段启发式规则，不能代替通用VAD。原 `results/followup_experiments/` 保留改动前结果，本轮诊断及170次音量/噪声回归写入 `tmp/registration_diagnosis/20261031/`。降低裁剪门槛虽可使5条本人录音全过，也会使另一现场用户录音通过，未采用。

### 注册与验证一致预处理实验（2026-10-31）

使用已有注册/验证临时WAV、两个现场用户档案及99条正式数据，首轮比较“原预处理重注册、仅声纹活动帧、仅口令连续活动区间、两者同时”；当前另加两组过滤孤立短活动的边界对照。每组同步重建码本、模板与阈值，固定归一化、特征与码本大小。现场档案从原模板恢复对应注册WAV；缺少这些本地输入时脚本会停止，不能把这组私人回放当作仓库自带数据。

```powershell
.\code\python\venv\Scripts\python.exe code/python/frame_consistency_experiments.py
# 仅运行两个现场用户的现有录音对照
.\code\python\venv\Scripts\python.exe code/python/frame_consistency_experiments.py --skip-formal --output-dir tmp/registration_diagnosis/live_only
# 用已保存参数及其对应预处理策略复核单条录音
.\code\python\venv\Scripts\python.exe code/python/frame_consistency_experiments.py --replay-candidate tmp/registration_diagnosis/20261031/sustained_preprocessing/candidates/lwl/both_sustained --wav tmp/gui_recordings/temp_verify_20261029_155706.wav
```

当前默认输出 `tmp/registration_diagnosis/20261031/sustained_preprocessing/`，包含5558次声明的逐条判定、汇总、相对重注册控制组的判定变化、背景帧/DTW局部代价归因、注册质量与整条留一自检，以及每组候选的参数和 `processing_policy.json`。首轮3754次结果保留在 `consistent_preprocessing/`。候选参数依赖配套预处理策略，**不能单独复制到App用户目录或作为默认模型加载**。活动标签仅是能量启发式筛选；口令区间连续保留内部停顿。

联合候选在5条本人录音上通过4条，现场其他人/循环背景对照未新增接受；剩余一条口令分数0.869686已过线，声纹1.048273仍被拒。但在12组“三条现有训练录音注册”的指定用户验证模拟中，本人通过22/72→26/72，误接受7/684→15/684，未达到部署要求。同一WAV反复对不同声明模型评分，不能当作独立新增录音，也不能与历史63条多用户识别准确率直接比较。

进一步检查发现，10–30ms的孤立活动点会把大段背景一起纳入口令边界。新增候选把相距不超过150ms的活动归组，忽略累计活动不足60ms的孤立组，保留内部停顿；同步重注册后现场本人又回到2/5通过，模拟注册联合候选为23/72通过、13/684误接受，仍未达到部署标准。

因此候选未替换默认模型及实际用户档案，当前App仍为2/5通过；现场两人的本人错误口令、真实麦克风验收仍缺。完整结果、背景归因及剩余任务见 `docs/结题改造主线.md` 顶部的当日记录。44项自动测试通过，不代表现场误拒已经解决。

### App注册质量检查

重启App后，注册样本列表和详情会显示有效活动时长、近满幅、孤立短促声音与首尾活动提示，并给出重录建议。可选中一条后点击“移除所选录音后重录”，其特征同步移除，原WAV保留；录音期间禁用。静音明确显示为不计入注册，有效持续活动不足0.20秒也不计入。其余新增提示是建议，不等于检测到错误口令；本地42条已有注册/训练录音的合格结果前后相同。

完成注册时，录音路径/内容校验、质量指标、是否参与建模及整条留一自检分数保存在 `app_data/diagnostics/enrollments/` 的独立JSON中，再保存用户模型。静音前置拒绝未产生的SNR/质量分写为null。自检只使用注册录音，不能保证独立验证通过；6条现场注册录音中3条检出孤立短活动，是否改善重新注册后的效果尚未验收。新提示使用能量规则，不修改既有用户档案或默认验证算法。

### 现场操作与删除用户（2026-10-31）

重启App后，注册录音、质量/显示特征计算、建模自检和验证匹配在后台运行。页面显示录音进度和电平，麦克风启动后再提示说口令；可点击“取消本次操作”。取消建模或匹配须等当前计算步骤结束，不保存迟到结果。操作期间锁定用户选择、删除、导航及模型/采样率设置；关闭窗口会先等待后台退出。麦克风测试也改为后台运行。

要重新注册：在“声音验证”选中用户，点击“删除所选用户”并确认，再点击“注册新用户”。删除只移除该注册模型，原录音和验证记录保留；操作期间禁用删除。也可以直接用同名重新注册，在完成时确认更新已有用户。

新一轮6次验证经用户标注为2条本人、2条其他人、2条遗忘；未知记录不用于统计。连同旧5条，当前确认本人通过2/7，仍有5条误拒。口令MFCC、MFCC+Delta、全部31维的固定对照均未提高这7条通过数，注册模拟的误接受还可能增加，因此未替换默认参数。详细结果和标签在忽略目录 `tmp/registration_diagnosis/20261031/new_session/`，不能把6次全拒绝说成6次本人误拒。

50项自动测试通过，含取消、关闭、删除后重新注册及原算法回归；真实麦克风下的新界面、设备释放和重新注册效果仍需实际验收。

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

使用流程：

1. 打开即进入“声音验证”。没有用户时点击“注册新用户”。
2. 在“注册用户”填写姓名，逐次录入至少三条合格的“芝麻开门”；进度条显示还差几条，达到三条后可选择继续补录。
3. 点击“完成注册，去验证”，应用自动建模、保存，并回到首页选中刚注册的用户；同名更新需要确认。
4. 在首页点击“开始录音验证”。页面显示是否通过和声纹/口令状态；需要解释时展开“查看分数与录音特征”。切换用户时自动加载，无须点击加载按钮。
5. 重启 App 后，已保存用户自动出现在首页，直接选择验证即可。开始注册另一个用户不会清掉当前验证用户；注册中途切换页面也保留已录样本。
6. “验证记录”只显示最近20条现场验证，包含时间、用户、通过/未通过、原因和耗时。记录自动保存，重启恢复；已有音频演示不混入这些记录。
7. “项目实验”包含“已有音频演示”和“算法对比结果”。前者用于固定WAV复现，后者用中文展示99条数据上的离线对比；刷新只是读取已有结果，不启动训练，也不代表当前注册用户的表现。

用户模型与归一化参数、质量基线、DTW 设置和阈值一起保存到 `app_data/users/`，验证使用该用户注册时的参数。本地用户和现场验证记录不计入正式实验数据，`app_data/` 已被 Git 忽略。

首次建立新用户模型仍需要随工程提供的基线模型和特征文件。用户注册、建模、保存、加载和验证均在 Python App 内完成，不需要 MATLAB。现场录音的正确口令、相近错误口令及陌生人拒识仍需在答辩设备上实际验收。

检查固定样例、注册建模、用户模型保存/加载后判定一致，以及历史保存/加载：

```powershell
.\run.ps1 gui --smoke
```

检查页面导航、注册自动保存、切换用户、注册进度保留、现场/演示记录隔离和录音失败后的状态恢复（使用已有WAV替代麦克风输入）：

```powershell
.\code\python\venv\Scripts\python.exe -m unittest discover -s code/python -p test_gui_workflow.py -v
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
