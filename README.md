# MFCC + VQ 声音锁

基于 Python 和 PySide6 的声纹口令验证课程项目。用户录制四条“芝麻开门”注册个人模型，验证时依次检查声纹、口令声学模板和识别文字。当前版本为 **v1.1.0**。

## Windows 下载

[下载 Windows x64 版](https://github.com/7M7666/voice-lock-mfcc-vq/releases/download/v1.1.0/SRDPVoiceLock-1.1.0-windows-x64.zip) · [发行说明与 SHA-256](https://github.com/7M7666/voice-lock-mfcc-vq/releases/tag/v1.1.0)

支持 Windows 10/11 64 位。完整解压 ZIP，双击 `SRDPVoiceLock.exe`，保留旁边的 `_internal` 文件夹。无需安装 Python 或 MATLAB，离线识别模型随包提供。

## 使用方法

1. 打开“注册用户”，输入名称，按提示录制四条合格的“芝麻开门”。不合格录音需补录；出现样本差异或覆盖提醒时，检查录音后确认是否继续。
2. 完成注册，在“声音验证”选择账号，用新录音验证。
3. 在“验证记录”查看、导出或复核最近 20 条记录，重启后保留用户和历史。

打包版用户数据保存在 `%LOCALAPPDATA%\SRDPVoiceLock`。更新版本时解压到新目录即可，用户数据独立保存。详细操作见 [Windows 启动说明](packaging/Windows启动说明.md)。

## 算法

| 环节 | 方法 |
|---|---|
| 声纹验证 | 提取 MFCC 特征，以 LBG-VQ 比较目标账号声纹 |
| 口令模板验证 | 以 MFCC + DTW 比较注册口令的声学模板 |
| 口令文字验证 | 以 sherpa-onnx Paraformer 离线识别，规范化后匹配“芝麻开门” |

三层全部通过才接受。ASR 模型缺失、校验失败或推理失败时拒绝放行。本项目为未接入实体门锁的课程原型，可能出现误拒和误接受。

## 源码运行

使用 Python 3.11，在 Windows PowerShell 中执行：

```powershell
git clone https://github.com/7M7666/voice-lock-mfcc-vq.git
cd voice-lock-mfcc-vq
python -m venv code/python/venv
.\code\python\venv\Scripts\python.exe -m pip install -r packaging/requirements-windows.txt
```

按 [离线语音识别说明](docs/离线语音识别接入说明.md) 准备指定版本的 `model.int8.onnx` 和 `tokens.txt`，放入 `models/asr/`，再启动：

```powershell
.\run.ps1 gui
```

仓库已包含基础模型、特征和 99 条正式录音，无需重新训练。源码运行产生的个人数据位于项目的 `app_data/` 和 `tmp/gui_recordings/`。

## 测试与打包

在项目根目录执行：

```powershell
.\code\python\venv\Scripts\python.exe -m unittest discover -s code/python -p 'test_*.py' -v
.\packaging\build_windows.ps1 -Version dev-local -Output tmp_release/new-local-build
```

打包需先准备 ASR 模型，输出目录必须不存在；可用 `-AsrModelDir` 指定模型目录。

## 项目目录

| 目录 | 内容 |
|---|---|
| [code/python/](code/python/) | GUI、算法与自动测试 |
| [data/demo_audio/](data/demo_audio/) | 正式录音与数据清单 |
| [models/](models/) / [data/features/](data/features/) | 模型与特征 |
| [packaging/](packaging/) | Windows 构建脚本与启动说明 |
| [results/](results/) | 实验结果，见 [结果说明](results/README_结果说明.md) |
| [docs/](docs/) | 开发记录、验收及模型来源与许可说明 |
| [code/matlab/](code/matlab/) / [code/legacy_notebooks/](code/legacy_notebooks/) | 早期实现 |
