# 中期答辩 PPT 制作说明

## 建议页数

10-12 页即可，不要堆太多代码截图。

## 页面结构

1. 封面：项目名称、成员、指导老师。
2. 研究背景：声纹认证、复杂声学环境、轻量化声音锁。
3. 研究目标：MFCC+LBG-VQ、多环境数据、拒识阈值。
4. 技术路线图：录音、预处理、特征提取、Z-score、LBG-VQ、阈值判定、结果输出。
5. 截至 5.20 的计划对照：已经完成什么，还缺什么。
6. 当前代码工程：展示 `srdp.py` 单文件主流程和 `run.ps1` 命令入口。
7. 数据采集规范：人数、环境、口令、metadata。
8. 阶段结果：MFCC 图、预处理图、环境特征对比图。
9. 识别与拒识实验：识别率表、混淆矩阵、`REJECT` 示例。
10. MATLAB UI 原型：界面截图和功能说明。
11. 存在问题：真实数据还需扩充、噪声环境覆盖有限、阈值需要更多样本校准。
12. 下一步计划：补齐数据、参数对比、完善 GUI、准备结题材料。

## 成员分工

- 冷雨涵：统筹 PPT 逻辑、开题/中期口径、最终修改。
- 李宛霖：提供 Python 流水线、实验结果图表、算法页讲解。
- 张新宇：提供 MATLAB UI 截图和 GUI 功能说明。
- 王天乐：提供数据采集表、录音样本数量统计、环境照片或说明。
- 隆高灿：负责文献综述、页面排版、答辩稿润色。

## PPT 必放文件

优先使用这些结果图表：

```text
results/figures/mfcc_single_user.png
results/figures/environment_feature_overview.png
results/figures/lbg_vq_confusion_matrix.png
results/tables/snr_estimates.csv
results/tables/feature_summary.csv
results/tables/lbg_vq_recognition_results.csv
```

注意：现在 demo 结果只能说明流程跑通。采集真实 SRDP 数据后，要重新运行：

```powershell
.\run.ps1 pipeline --metadata data/demo_audio/metadata_srdp.csv
```

然后再把新图表放进 PPT。

## 答辩口径

可以这样讲：

“截至中期，项目已完成从语音采集、预处理、多维声学特征提取、Z-score 归一化，到严格 LBG-VQ 码本训练和拒识阈值判定的完整实验流水线。当前 demo 数据用于验证流程可运行，下一步重点是按统一规范补齐多人多场景自采数据，并基于真实数据输出识别率、拒识率和环境鲁棒性对比结果。”

