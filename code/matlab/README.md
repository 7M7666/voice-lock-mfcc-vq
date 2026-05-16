# MATLAB 代码说明

正式答辩演示入口在 `app/`：

```text
app/run_voice_lock_app.m
app/SRDPVoiceLockAppFixed.m
```

`legacy_app/` 保存组员原始 App Designer 文件和旧 `.m` 源码，只作留档。

`legacy_preprocess/` 保存早期 MATLAB 预处理实验脚本。当前正式识别、特征提取、模型训练和结果生成均以后端 `code/python/srdp.py` 为准。
