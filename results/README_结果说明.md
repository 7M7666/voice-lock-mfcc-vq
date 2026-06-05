# 结果文件说明

本目录保存正式 SRDP 自采数据跑出的中期答辩结果。

## 表格

- `tables/snr_estimates.csv`：每条录音的近似 SNR 估计。
- `tables/feature_summary.csv`：每条录音的特征统计摘要。
- `tables/lbg_vq_recognition_results.csv`：逐条样本的声纹判定、口令验证和最终判定。
- `tables/voice_lock_metrics.csv`：单层声纹识别和两层声音锁的汇总指标。
- `tables/experiment_matrix.csv`：特征组合与 codebook size 的批量对比结果。
- `tables/dtw_strategy_matrix.csv`：不同口令 DTW 策略的对比结果。
- `tables/fusion_matrix.csv`：融合权重、融合阈值和音频质量权重的扫描结果。
- `tables/recommended_config.csv`：按安全优先规则筛出的结题推荐配置。

## 图表

- `figures/environment_feature_overview.png`：不同环境下的特征概览。
- `figures/snr_by_environment.png`：不同环境 SNR 对比。
- `figures/accuracy_by_split.png`：不同测试类型的准确率对比。
- `figures/lbg_vq_confusion_matrix.png`：两层声音锁最终混淆矩阵。
- `figures/lbg_vq_voice_only_confusion_matrix.png`：单层声纹识别混淆矩阵。
- `figures/passphrase_dtw_distance.png`：口令 DTW 距离分布。
- `figures/error_breakdown.png`：主要错误类型统计。
- `figures/two_layer_pipeline.png`：两层验证流程图。

重新生成这些结果：

```powershell
.\run.ps1 pipeline --metadata data/demo_audio/metadata_srdp.csv
```
