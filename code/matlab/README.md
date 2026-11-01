# MATLAB 代码说明

`app/` 是中期 MATLAB 展示入口，现仅保留为历史版本；结项正式入口为 `code/python/gui.py`（项目根目录执行 `.\run.ps1 gui`），见[项目 README](../../README.md)。本机 Python App 直接计算注册和验证，不依赖 MATLAB。

历史 MATLAB 入口：

```text
app/run_voice_lock_app.m
app/SRDPVoiceLockAppFixed.m
```

`legacy_app/` 保存组员原始 App Designer 文件和旧 `.m` 源码，只作留档。

`legacy_preprocess/` 保存早期 MATLAB 预处理实验脚本。算法实现保留在 `code/python/srdp.py`；结项沿用现有模型和结果，不重训、不重跑生成流水线。
