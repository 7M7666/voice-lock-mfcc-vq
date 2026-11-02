# Windows 发行准备验收（2026-11-02）

**结论：完成源码检查、必要修复、文档整理和本机 Windows 构建/离线自动验证；尚不具备正式公开发行条件。** 模型完整再分发许可、干净 Windows、真人独立注册及冒充验收仍待完成。没有创建新 Release，也没有上传权重或本地安装包。

## 基线与分支

- 开始前执行 `git fetch origin --prune`，远程 master 为 `2430a405cc6ce06660a22de2e2a833535b2f1bef`，真实包含 E 个人验证代码。本地原 master 为 `eb8b772`，没有未提交文件；本轮没有在 master 开发。
- 从 origin/master 创建 `chore/release-readiness-docs`。本地另有两条 frozen-round2 feature 分支，远程还保留 E feature、两条 frozen-round2 feature 及五条 experiment 分支；均未覆盖、删除或强推。
- 五条实验分支为 `experiment/mfcc-dtw-public`、`experiment/field-reliability-public`、`experiment/ecapa-offline-asr`、`experiment/ecapa-asr-analysis`、`experiment/consistency-aishell`。ECAPA 未部署。
- 当前 v1.0.0 tag 的 GUI 没有 ASR，是旧双层版本；不能把本轮构建或 master 功能写成已发布。

## 已完成的检查与修改

| 文件/范围 | 检查结果与最小修改 |
|---|---|
| `code/python/gui.py`、`asr_phrase.py` | 以异常类型区分录音、ASR 与其他计算失败，后台保留异常对象；ASR 失败不新增成功历史。同步 WAV 验证先清除旧结果；历史复核捕获 ASR 错误并保留原记录 |
| `acceptance_baseline.py`、`frozen_round2_eval.py`、`phrase_content_experiments.py`、`live_fusion_experiments.py`、`frame_consistency_experiments.py` | 修复历史工具悄悄使用 E 的回归，显式调用 `predict_with_session_model_baseline`，保持 A / 冻结比较协议；本轮未运行参数实验 |
| 相关 `test_*.py` | 新增加载/推理错误、错误文字、空转写、路径覆盖、旧历史、GUI 恢复和历史入口不调用 ASR 等测试；冻结与校准测试改为核对 A |
| `README.md` | 从 275 行整理为 91 行：简介、旧版下载、源码运行、四条注册、核心算法、完成状态与文档入口；移除大量历史表格和永久停止研究式表述 |
| `docs/`、数据/结果/旧代码 README | 更新 E 已合并状态，修复不存在的采集规范和旧 README 锚点，旧计划原地标为归档；未移动或删除研究资料 |
| `packaging/build_windows.ps1` | 默认版本改为 `dev-local`，避免新版误标 v1.0.0 |
| `packaging/make_release.py` | 加强私人目录审计、标明本地验收状态，许可证收集不复制 Python 源码/字节码和缓存。首次 onedir 的许可目录曾混入缓存和本机路径，修复后在全新目录组装并复查 |
| `.gitignore` | 补充通用 venv、环境变量文件和 ONNX 权重忽略；原现场数据与临时目录仍受保护 |

`code/python/srdp.py`、`speech_frames.py`、`dtw_paths.py`、基础模型和正式阈值未修改。对 E 之前 `eb8b772` 的 GUI AST 比较确认：原预测函数（仅改名称为 baseline）、注册质量、注册建模、一致性与自检、用户模型存取实现均一致；四条数量与质量常量未更改。

## 已验证的自动测试

实际输出存于 [results/release_readiness/2026-11-02/](../results/release_readiness/2026-11-02/)，日志仅将本仓库绝对路径替换为 `<REPO>`；没有现场用户或私人 WAV。

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 安装锁定 Windows 依赖、`pip check` | 安装 sherpa-onnx/core 2.0.1；无依赖冲突；Windows/Python 3.11.9 | [pip-check](../results/release_readiness/2026-11-02/pip-check.txt)，完整安装日志本机留存 |
| 全套 `unittest discover` | **131 项通过** | [source-tests](../results/release_readiness/2026-11-02/source-tests.txt) |
| 实际 EXE `--self-test` | **49 项通过，退出码 0**；末尾 `Bundled ASR inference: passed` | [exe-self-test](../results/release_readiness/2026-11-02/exe-self-test.txt) |
| 真实 ASR，本地指定模型 | 正确样例“芝麻开门”；错误样例“快快开门” | [offline-asr](../results/release_readiness/2026-11-02/offline-asr.txt) |
| E 接受是 A 子集 | 252 次固定账号声明，A 接受20，E接受19，新增接受0；各次 VQ/DTW 距离、分数与阈值一致 | [regression](../results/release_readiness/2026-11-02/regression.json) |
| 旧格式模型、新进程重新加载 | 从基础模型生成省略可选字段的旧格式档案，重启 Python 进程后，同 WAV 完整结果字典一致；不是读取私人旧档案冒充真人验收 | [输出](../results/release_readiness/2026-11-02/regression.txt)、[检查脚本](../results/release_readiness/2026-11-02/regression_check.py) |
| 四条准入、旧历史与 GUI 恢复 | 三条不能注册、四条可以；不合格不计数；正常拒绝保存；ASR/麦克风失败不沿用旧成功，取消/重启恢复通过模拟测试 | 同上自动日志；GUI 的 ASR 流程测试有替身，真实模型另行推理验证 |

全套命令：

```powershell
$env:PYTHONUTF8 = '1'
.\code\python\venv\Scripts\python.exe -m unittest discover -s code/python -p 'test_*.py' -v
.\code\python\venv\Scripts\python.exe results/release_readiness/2026-11-02/regression_check.py
```

没有执行会覆盖历史模型/表格的 `pipeline`、`smoke-test --skip-pipeline`，没有 DTW/MFCC/ECAPA/AISHELL 新参数实验，没有降低阈值。

## 历史基线与本次回归

历史自动识别协议为 36 条训练、63 条评估、先选最近说话人再拒识，双层 56/63；与个人声明账号不是同一任务。更详细历史数据保留在[结果说明](../results/README_结果说明.md)及原表。

本次使用同样 63 条评估 WAV、四个旧账号模型，真实 ASR 配对检查 A/E，共252次声明：

| 类别 | 声明次数 | A 接受 | E 接受 |
|---|---:|---:|---:|
| 本人正确口令 | 24 | 17 | 17 |
| 本人错误口令 | 36 | 1 | 0 |
| 他人正确口令 | 84 | 2 | 2 |
| 他人错误口令 | 108 | 0 | 0 |

E 仅新增拒绝 `S02_wrong/dorm/S02_wrong_003.wav` 对 S02 的声明。原有两次“他人正确口令”误接受仍存在。语音文字检查不能代替身份验证；少量数据上的零误通过不代表总体安全。数据已多次参与开发，多账号声明不是独立新录音，以上不是现场新用户或盲测通过率。

## Windows 构建与资源核查

在本机 Windows x64 环境执行：

```powershell
.\packaging\build_windows.ps1 -Version e-readiness-local -Output tmp_release/readiness-20261102/build-e
```

PyInstaller 6.22.0 onedir 构建成功，包含 sherpa 的 ONNX Runtime/C/C++ DLL、Qt `qwindows.dll`、Python DLL、指定哈希 ASR 两文件、原基础模型/特征、有限表格和9条仓库样例。通过清除开发 PATH、从 Windows 目录启动的 EXE 自检。`voice_lock.spec` 的模型哈希、DLL来源约束和资源路径均保留；`windows_entry.py` 的真实模型自检已执行，无需改写。依赖文件未调版本。

首次构建后发现许可收集复制了 `packaging/licenses` 中的缓存。修复过滤后复制到全新 `audited-dist`（保留首次构建），重新执行 `make_release.py`，最终本地包为：

- `tmp_release/readiness-20261102/audited-dist/SRDPVoiceLock/`
- `tmp_release/readiness-20261102/audited-dist/SRDPVoiceLock-e-readiness-local-audited-windows-x64.zip`
- 大小 **244350445 bytes**；SHA-256 **`a7ce234640aa8a998446598a2297a2dcf02904b877c4067e74f3fd0f2b3abfd2`**。
- 最终1053项清单哈希通过，ZIP共1054文件；没有私人用户目录、录音、历史、临时缓存。扫描应用可执行/文本载荷未见本仓库或用户目录绝对路径。最终 EXE 字节与已通过49项自检的 EXE 相同。
- 构建时工作树尚未提交，`build-info.json` 如实标记基线提交和 `source_tree_dirty=true`，并记录源文件哈希；这是本地验收产物，不是干净提交的正式发行证据。
- 日志有 `pycparser.lextab/yacctab`、`scipy.special._cdflib` 可选隐藏导入和 Numba `tbb12.dll` 未找到警告。当前代码未使用 `parallel=True` / `prange`，本轮算法及EXE测试未触发这些可选路径；未宣称所有第三方可选功能已验证。

资源证据：[原资源审计](../results/release_readiness/2026-11-02/resource-audit.txt)、[最终包审计](../results/release_readiness/2026-11-02/final-package-audit.txt)。544份原有非文档 data/models/results 文件与远程基线 Git 内容一致，仍有99条正式 WAV；改动只涉及其说明与新增验收目录。

## 实际窗口检查与未验证项

实际 EXE 已在隔离 LOCALAPPDATA 中启动；可访问性树读到“模型: 已加载”、S01 测试账号与一条固定WAV历史；第一次正常关闭后又启动了同一 EXE。数据由仓库基础模型/固定 WAV 生成，未操作真实个人档案。

桌面截图报 `no monitor found for window`，坐标操作报 `coordinate input geometry is unavailable`，因此**没有完成视觉布局验收**。用户随后按 Escape 停止 Computer Use，并明确要求不要控制电脑；已停止全部桌面控制。第二次启动后的交互/退出验收未完成，不将首次可访问性读取或自动测试夸大为完整实机操作验收。[已取得的可访问性证据](../results/release_readiness/2026-11-02/exe-window-accessibility.txt)保留，未截取或发布用户桌面。

## 模型许可与公开发布限制

框架 LICENSE 为 Apache-2.0；确切模型来源链已核对到 HF → crazyant ONNX量化 → damo 原模型，后者 README 元数据标注 Apache License 2.0。HF卡片仅列来源，crazyant/damo 独立 LICENSE 请求404，未补齐绑定本次转换权重、tokens和相关NOTICE的完整再分发依据。结论仍是**完整权重再分发许可未确认**，不是根据404认定禁止使用。

来源、哈希与限制详见[离线识别说明](离线语音识别接入说明.md#许可核查2026-11-02)。模型只存 Git 忽略目录，两个本地 ZIP 均未上传，不创建或发布新 Release。

## 待完成的真人与实机验收

- [ ] 在干净 Windows 10/11 x64 电脑、无开发环境、完整解压和中文路径下启动、关闭重开，人工检查窗口布局、模型、历史、取消及异常恢复；当前机器截图不可用的部分亦待检查。
- [ ] 新用户现场取得四条原质量门槛合格录音；三条不能注册、不合格不计数；冻结模型后再录独立测试 WAV。
- [ ] 本人正确、本人相似错误、他人正确、他人错误四类测试，预先记录口令原文、目标账号、说话人、设备、距离、次数，保留全部成功与失败；建议每类至少3次用于流程验收，少量次数不能估计广泛安全性。
- [ ] 真实麦克风权限、拔插/占用、静音、连续验证、取消和关闭重启；ASR缺失/错误时提示准确且不放行。保存WAV、模型指纹和导出CSV，CSV不代替音频，App仅保留最近20条记录。
- [ ] 完成权重再分发依据和依赖许可证/NOTICE核对后，从可追溯干净提交重新构建并在干净电脑验收；满足这些条件前不发布新 Windows Release。

现场验收不得通过调阈值、改质量准入、删除失败录音或挑选成功样本制造通过率。本轮仅准备与核验现有实现，不新增算法研究。
