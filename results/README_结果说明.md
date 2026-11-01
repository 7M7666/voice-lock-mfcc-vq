# 结果文件说明

本目录保存正式 SRDP 自采数据跑出的中期答辩结果。

**2026-11-01 结项说明**：本目录原表、图和模型引用原样保留，不重跑或覆盖。`tables/voice_lock_metrics.csv` 的 56/63 对应保存的旧版四人模型与 36/63 训练/评估划分，先自动选择最接近的说话人，再做拒识和 DTW 模板匹配；它不是指定账号认证或现场新用户成绩。“错误口令36/36、陌生人3/3拒绝”也仅适用于该自动识别协议。当前训练实现已经修复，保存码本仍来自旧实现，不能将重新训练视为同一历史模型。

指定账号固定回归使用相同 63 条 WAV，分别验证四个旧账号，共 252 次声明：本人正确 17/24 通过、本人错误 1/36 误通过、他人正确 2/84 误通过、他人错误 0/108 误通过。重复账号声明不是新增独立录音；错误口令缺少逐字标签。数据已经参与开发，不能作为新盲测。入口与本机结果位置见[项目 README](../README.md#指定账号的固定回归)。

`experiments/`、`dtw_experiments/`、`fusion_experiments/`、`calibration_experiments/` 和 `followup_experiments/` 等目录保存各轮离线对照，只能在其模型、特征、划分和决策协议下解释。`recommended_config.csv` 是已用数据上筛出的历史候选，不是 App 当前配置或最终安全性结论。私人录音实验留在本机临时目录，部分原始录音已缺失；不能据历史汇总表声称本次重新验收成功。真实设备四类验证由用户按 README 清单执行并单独统计。

## 表格

- `tables/snr_estimates.csv`：每条录音的近似 SNR 估计。
- `tables/feature_summary.csv`：每条录音的特征统计摘要。
- `tables/lbg_vq_recognition_results.csv`：逐条样本的声纹判定、口令验证和最终判定。
- `tables/voice_lock_metrics.csv`：单层声纹识别和两层声音锁的汇总指标。
- `tables/experiment_matrix.csv`：特征组合与 codebook size 的批量对比结果。
- `tables/dtw_strategy_matrix.csv`：不同口令 DTW 策略的对比结果。
- `tables/fusion_matrix.csv`：融合权重、融合阈值和音频质量权重的扫描结果。
- `tables/recommended_config.csv`：当时按安全优先规则筛出的离线候选，未作为结项配置部署。

## 图表

- `figures/environment_feature_overview.png`：不同环境下的特征概览。
- `figures/snr_by_environment.png`：不同环境 SNR 对比。
- `figures/accuracy_by_split.png`：不同测试类型的准确率对比。
- `figures/lbg_vq_confusion_matrix.png`：两层声音锁最终混淆矩阵。
- `figures/lbg_vq_voice_only_confusion_matrix.png`：单层声纹识别混淆矩阵。
- `figures/passphrase_dtw_distance.png`：口令 DTW 距离分布。
- `figures/error_breakdown.png`：主要错误类型统计。
- `figures/two_layer_pipeline.png`：两层验证流程图。

以下命令仅保留为历史生成入口，本次结项不执行；它会覆盖默认特征、模型和部分结果，当前训练实现也可能产生不同成绩：

```powershell
.\run.ps1 pipeline --metadata data/demo_audio/metadata_srdp.csv
```

`smoke-test --skip-pipeline` 虽跳过训练，仍会重写推荐配置表，也不是只读检查。运行 App 和查看已有表无需执行上述命令。
