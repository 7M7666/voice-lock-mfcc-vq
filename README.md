# MFCC + VQ 声音锁

在普通 Windows 电脑上，通过四条合格录音注册个人声音模型，再验证声纹与固定口令“芝麻开门”。主程序为 Python + PySide6 软件原型，未接入实体门锁。

## 版本与完成状态

截至 **2026-11-02**，请区分以下版本：

| 对象 | 实际状态 |
|---|---|
| [v1.0.0 Windows Release](https://github.com/7M7666/voice-lock-mfcc-vq/releases/tag/v1.0.0) | 已发布旧版，MFCC + LBG-VQ 声纹、MFCC + DTW 口令模板双层验证；**不含 E / ASR** |
| 当前 `master` | 已接入 E：原双层接受后，再由 sherpa-onnx Paraformer 离线检查文字；不是已发布 Windows 版本 |
| 新 Windows 版本 | 发行准备和本机构建验证见[验收记录](docs/Windows发行准备验收.md)；尚未正式发布，仍需干净电脑与真人独立验收、权重许可确认 |
| ECAPA 等候选 | 保存在实验分支，未部署到正式 App；本轮不研究或调参 |

已实现注册质量检查、个人模型保存加载、验证历史、取消和复核。**新用户现场注册、冒充测试、跨设备及回放防护没有充分的独立验收证据**，不能将课程原型视为可靠门禁产品，也不编造通过率。

## Windows 下载

下载 [v1.0.0 Windows x64 ZIP](https://github.com/7M7666/voice-lock-mfcc-vq/releases/download/v1.0.0/SRDPVoiceLock-1.0.0-windows-x64.zip)，完整解压后双击 `SRDPVoiceLock.exe`；保留旁边的 `_internal`。无需 Python 或 MATLAB。Release 页的 Source code 是源码归档。

这是**旧双层版本**。本仓库当前的[Windows 启动说明](packaging/Windows启动说明.md)描述正在准备的含 E 版本，不表示下载的 v1.0.0 已含 ASR。没有发布新版之前，不将本地验收包作为公开发行包传播。

## 源码运行

Windows PowerShell，Python 3.11，在项目根目录执行：

```powershell
git clone https://github.com/7M7666/voice-lock-mfcc-vq.git
cd voice-lock-mfcc-vq
python -m venv code/python/venv
.\code\python\venv\Scripts\python.exe -m pip install -r packaging/requirements-windows.txt
.\code\python\venv\Scripts\python.exe -m pip check
```

Windows 构建依赖有版本锁定；[源码依赖](code/python/requirements.txt)仅固定 ASR 版本，其余未全量锁定。仓库已含基础模型、特征和 99 条正式 WAV，启动无需重训。

个人验证还需自行准备指定的 `model.int8.onnx` 和 `tokens.txt`，放在 `models/asr/`，或设置 `$env:VOICE_LOCK_ASR_DIR` 指向模型目录。[ASR 说明](docs/离线语音识别接入说明.md)列出来源、SHA-256、路径和许可边界。程序不自动下载；文件缺失、校验失败或推理失败不会放行。

```powershell
.\run.ps1 gui
# 或
.\code\python\venv\Scripts\python.exe code/python/gui.py
```

## 四条注册与个人验证

1. 在“注册用户”输入名称，每条自然说出“芝麻开门”，录音首尾留少量空白。
2. 必须取得 **4 条合格录音**：总长 1.2–5 秒、持续有效声音至少 0.65 秒、估计 SNR 至少 10 dB、质量评分小于 0.35，且无削波或首尾截断标记。不合格录音不计数，可移除后补录。
3. 四条后完成注册；一致性异常会提示复核。保存个人模型并返回“声音验证”。注册自检只使用注册样本，不能算独立认证验收。
4. 选择账号，用新录音验证。先通过 VQ 与 DTW，再要求 ASR 规范化转写严格等于“芝麻开门”。错误文字正常拒绝；麦克风错误与 ASR 错误分别提示，错误操作不新增成功历史。
5. 在“验证记录”查看最近 20 条个人验证、导出或复核。退出重启后恢复用户和历史；旧历史缺少的 ASR 字段留空。旧记录复核使用当前算法，不改写原记录。

源码数据位于 `app_data/users/`、`app_data/history.csv`、`app_data/diagnostics/` 和 `tmp/gui_recordings/`。打包版对应目录位于 `%LOCALAPPDATA%\SRDPVoiceLock`，与安装资源分离。CSV 导出不包含 WAV，验收时另存原音频与标签；这些私人数据均不得提交或装入发行包。

## 核心算法与证据边界

- **A 原算法**：默认 12 维 MFCC；LBG-VQ 比较目标账号声纹，MFCC + DTW 比较口令声学模板。DTW 不是文字识别。当前基础码本及阈值保留；2026-10-29 修复的 LBG 训练实现只影响之后新建模型，不能把重训当成复现旧码本。
- **E 个人验证**：仅在 A 接受后串联离线 Paraformer；按 NFKC、去空白与标点/符号后精确匹配“芝麻开门”。E 接受必须是 A 接受的子集，不修改 VQ/DTW 参数或放宽质量门槛。ASR 可能增加误拒，不能消除声纹冒充风险。
- **历史离线入口**：`srdp.py predict-one`、项目实验演示和已有实验工具保持 A / 各自冻结协议，不自动使用 E。它们的旧指标不是三层个人验证或新用户现场成绩。

99 条正式录音及历史数据划分、56/63 自动识别成绩、252 次账号声明回归等详细依据见[结果说明](results/README_结果说明.md)与[历史开发主线](docs/结题改造主线.md)。已有数据多次参与开发；多账号声明不是新增独立录音，小样本零误通过不证明总体风险为零。

## 测试与 Windows 构建

```powershell
.\code\python\venv\Scripts\python.exe -m unittest discover -s code/python -p 'test_*.py' -v
# 以下只在本机生成含模型的验收包；许可未确认前禁止公开上传
.\packaging\build_windows.ps1 -Version dev-local -Output tmp_release/new-local-build
```

输出目录必须不存在，可用 `-AsrModelDir` 指定构建模型目录。构建校验模型哈希、收集 DLL 和有限资源，生成 onedir、文件清单和本地 ZIP；**构建成功不代表发布获准**。完整结果、EXE 自检及尚缺的实机/真人项目见[Windows 发行准备验收](docs/Windows发行准备验收.md)。

`pipeline` 会覆盖特征、基础模型和结果；`smoke-test --skip-pipeline` 仍会改写推荐配置表，二者均不是只读启动检查，本轮未执行。历史实验入口不是新用户使用步骤，部分依赖的私人 WAV 已缺失，不承诺全部可重放。

## 文档与目录

| 路径 | 用途 |
|---|---|
| [code/python/gui.py](code/python/gui.py) | PySide6 注册、个人三层验证与历史；[asr_phrase.py](code/python/asr_phrase.py) 提供 E |
| [code/python/srdp.py](code/python/srdp.py) | 原特征、VQ、DTW 和历史离线 CLI |
| [data/demo_audio/](data/demo_audio/) | 99 条正式录音与 [metadata](data/demo_audio/metadata_srdp.csv)，不混入现场验收录音 |
| [models/](models/) / [data/features/](data/features/) | 基础及历史模型、特征；`models/asr/` 为忽略的本地权重目录 |
| [results/](results/) | 保留的历史表图与本次验收输出，解释见[结果说明](results/README_结果说明.md) |
| [packaging/](packaging/) | Windows 构建、资源审计、EXE 启动及自检 |
| [docs/离线语音识别接入说明.md](docs/离线语音识别接入说明.md) | E 的调用边界、模型路径、哈希与许可来源 |
| [docs/结题改造主线.md](docs/结题改造主线.md) | 历史开发和实验记录；旧任务不自动成为当前要求 |
| [docs/结题分工/](docs/结题分工/) / [docs/archive/](docs/archive/) | 历史分工、计划和早期资料，保留备查 |
| [code/matlab/](code/matlab/) / [code/legacy_notebooks/](code/legacy_notebooks/) | 早期 MATLAB 与 Live Script，当前启动不依赖它们 |

五条 `experiment/*` 远程分支保留 MFCC/DTW、现场可靠性、ECAPA/ASR 和一致性研究；它们不是正式 App 的已部署功能。后续研究与本轮发行准备分开记录。
