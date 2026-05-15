### 项目结构

```
HAM10000_BIP_Classification/
│
├── main.py                          # 主入口：加载图像 → 完整pipeline → 输出诊断报告
├── config.py                        # 全局参数/阈值配置
│
├── preprocessing/
│   ├── __init__.py
│   ├── hair_removal.py              # 去毛发（黑帽变换 + inpainting）
│   ├── color_normalization.py       # 颜色校正（Shades-of-Gray）
│   ├── illumination_correction.py   # 光照均匀化（低频场估计 + 除法校正）
│   └── segmentation.py             # 病灶分割（Otsu + 形态学精修），输出mask和边界
│
├── features/
│   ├── __init__.py
│   │
│   ├── # ---- 第1层：组织来源判定特征 ----
│   ├── pigment_network.py           # 色素网络检测与规则性评估
│   │                                #   → has_network, regularity_score, coverage
│   ├── lacunae.py                   # 红色腔隙检测（血管来源）
│   │                                #   → has_lacunae, count, area_ratio, color_type
│   ├── central_white_patch.py       # 中央白斑+周围网络（纤维来源）
│   │                                #   → df_pattern_score, radial_gradient
│   │
│   ├── # ---- 第2层-黑色素细胞组特征 ----
│   ├── symmetry.py                  # 结构/颜色对称性
│   │                                #   → symmetry_score (0-2)
│   ├── color_variegation.py         # 颜色种类与分布
│   │                                #   → color_count, color_entropy
│   ├── blue_white_veil.py           # 蓝白幕检测
│   │                                #   → bwv_score, area_ratio
│   ├── streaks.py                   # 条纹/假足检测（极坐标展开 + 方向梯度）
│   │                                #   → streak_score, asymmetry_index
│   ├── dots_globules.py             # 不规则点/球检测（LoG多尺度）
│   │                                #   → irregularity_score, size_cv, distribution_entropy
│   ├── regression.py                # 回归结构（白色瘢痕 + 蓝灰颗粒）
│   │                                #   → regression_score, area_ratio
│   ├── vascular_pattern.py          # 血管形态分析（点状/线状/多形性）
│   │                                #   → polymorphism_score, morphology_type
│   │
│   ├── # ---- 第2层-角质形成细胞组特征 ----
│   ├── milia_cysts.py               # 粟粒样囊肿（白帽变换 + 圆形度）
│   │                                #   → milia_score, count
│   ├── comedo_openings.py           # 粉刺样开口（黑帽变换 + 圆形度）
│   │                                #   → comedo_score, count
│   ├── fissures_ridges.py           # 脑回样裂隙（Gabor + 曲率）
│   │                                #   → cerebriform_score
│   ├── border_sharpness.py          # 边界锐利度
│   │                                #   → sharpness_score
│   ├── strawberry_pattern.py        # 草莓样模式（红色背景 + 白色毛囊口）
│   │                                #   → strawberry_score
│   ├── surface_texture.py           # 表面鳞屑/粗糙度（小波高频能量）
│   │                                #   → roughness_score, scale_area_ratio
│   ├── arborizing_vessels.py        # 树枝状血管（Frangi + 骨架分支）
│   │                                #   → arborizing_score, branching_factor
│   ├── ovoid_nests.py               # 蓝灰卵圆巢
│   │                                #   → nest_score, count
│   └── leaf_spoke_structures.py     # 叶状/轮辐结构
│                                    #   → leaf_score, spoke_score
│
├── classification/
│   ├── __init__.py
│   ├── layer1_origin.py             # 第1层决策：组织来源分组
│   │                                #   输入：L1特征字典 → 输出：origin_group + 置信度
│   ├── layer2_melanocytic.py        # 第2层：nv vs mel（七点评分逻辑）
│   │                                #   输入：L2黑色素特征 → 输出：诊断 + mel_score
│   ├── layer2_keratinocytic.py      # 第2层：bkl vs akiec vs bcc（加权评分）
│   │                                #   输入：L2角质特征 → 输出：诊断 + 各类得分
│   └── pipeline.py                  # 串联完整三层决策流程
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                   # 准确率/召回率/混淆矩阵/per-class F1
│   └── report_generator.py          # 生成单图诊断报告（路径 + 各特征得分 + 判据）
│
└── utils/
    ├── __init__.py
    ├── color_spaces.py              # RGB↔Lab↔HSV转换、6色空间映射
    ├── morphology.py                # 通用形态学操作封装
    ├── gabor_bank.py                # Gabor滤波器组构建
    ├── polar_transform.py           # 极坐标展开/逆变换
    └── visualization.py             # 特征可视化（叠加标注到原图）
```
---
