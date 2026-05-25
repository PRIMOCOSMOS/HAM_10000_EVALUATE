# HAM10000_EVALUATE Refactored

本项目内容旨在使用传统方法完成HAM10000数据集的七分类任务，主要方法是使用BIP中的各类特征并进行轻量的机器学习分类.

## Main Pipeline

1. 预处理阶段：本项目内置了Black-hat形态学变换去毛发的算法，并通过CLAHE处理对比度、光照问题；但也可以直接通过参数设置，
跳过预处理，直接使用Enhanced数据（按照分工的原则），处理见前一位同学的工作；ROI根据普遍研究中使用的方法，使用Chan-Vese active contour.
2. 特征阶段：本项目使用了众多的图像的特征，并将其结果视作向量进行拼接，从而构造多特征联合分类任务；
3. 分类任务：利用机器学习进行分类，主要方法有SVM和ANN。在样本设定上，我们可以选择全数据集训练+验证，也可以像某些文件中所做的那样，进行均衡化采样。


## 技术细节

本项目一个最大的障碍就是样本的**极不均衡**，nv占到全样本数的67%，这不利于样本的分类，但由于传统方法的上限就严重受限，其具体造成的影响水平也存疑。本项目中我们采用SMOTE及其衍生方法加以应对，也就是通过在已有的少数类样本之间进行“插值”，生成全新的、合成的少数类样本，这本质上是一种数据增强算法。

### 特征选取

1. GLCM: 着重提取ROI的纹理特征；
2. LBP on ROI and BIMF1: 也是聚焦于纹理的特征；BIMF1是二维经验模态分解得到的第一个图层，包含了最为高频细致的成分，可以
提升LBP（局部二值模式）的敏感度；
3. HSV histogram: **色彩空间（Color Space）**特征，颜色直方图可以反映ROI的颜色倾向，这对应了医学上一些可解释的特征；
4. ABCD: 不对称性、边缘不规则、颜色不均/多样性、直径大小，直接对应医学诊断流程。但是这些特征高度概括而主观，量化设计比较困难，
尤其是直径大小难以直接量化，受拍摄条件和图像缩放影响。

LBP和HSV都有一定抗亮度变化的能力。


## 结果评估

虽然在这个项目中，我们使用了很多的特征和方法，但并没有一种方法组合展现出明显的优越性。事实上，一些文献采用了均衡化子集的方式进行训练，声称取得了比较好的结果，但对于方法细节讲得并不算详细，实践上也难以复现。

对于全数据集分类、验证任务，正确率大致在0.6-0.7之间；artifacts_*文件夹是结果输出，其中有混淆矩阵展示。ANN和SVM效果没有显著的差距，在目前的正确度水平下，这样的差距也缺乏意义。此外，对于数据集中的少数样本，分类器也很难建立足够的敏感性，即使是Smote及其衍生方法也没有完全解决这个问题。这和学术界目前的观点相符，也就是传统方法对于这类分类任务难有理想的效果。



## 项目结构

```text
HAM_10000_EVALUATE_refactored/
├── main.py
├── requirements.txt
└── ham_pipeline/
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── preprocess.py
    ├── segment.py
    ├── features.py
    ├── balancing.py
    ├── model.py
    ├── evaluate.py
    └── trainer.py
```

## 输出结构

每一轮测试分类都会产生一个`artifacts_*/`文件夹，其中单次测试的文件架构如下：

```
/artifacts_*/
    ├── class_distribution_train.csv 训练集及其类型分布
    ├── class_distribution_val.csv 测试集及其类型分布
    ├── classification_report 结果报告，比较晦涩，不如混淆矩阵直观
    ├── confusion_matrix.png 混淆矩阵（重点关注）
    ├── feature_columns.json 用于分类的特征向量拼接结果
    ├── feature_columns.json 标签分类
    ├── metrics.json 评估参数，里面有准确度
    ├── run_config 关键，有本次测试的一些选项
    └── model.joblib 分类器的模型文件
```

### run_config.json

```

run_config.json 记录本次训练/评估的完整运行配置，常见字段含义如下：

- dataset_root: 数据集根目录，如果用的是HAM10000_NOV就是前一位同学的Enhanced数据，HAM10000则是项目自己内置的raw_data；
- output_dir: 输出目录
- image_size: 输入图像缩放大小
- val_ratio: 验证集比例（split_mode=stratified_ratio 时生效）
- random_state: 随机种子，训练是随机抽取会用到
- n_jobs: 并行度，性能问题，不用关注
- cv_iter/cv_lambda1/cv_lambda2: Chan-Vese 分割迭代次数与权重，ROI。
- clahe_clip_limit/clahe_tile_grid_size: CLAHE 参数 - 内置的预处理，已有预处理数据则跳过
- blackhat_kernel_size/inpaint_radius: 去毛发参数 - 内置的预处理，已有预处理数据则跳过
- preprocessing_enabled/enable_hair_removal/enable_clahe: 预处理开关，布尔变量
- lbp_points/lbp_radius/lbp_method/lbp_bins: LBP 参数
- include_lbp_roi/include_lbp_bimf1/include_glcm/include_hsv/include_abcd: 特征开关，这些特征可以选择是否参与
- mremd_window_size/mremd_smoothing_kernel/mremd_max_bimfs: MREMD 参数，计算BIMF1会用到
- hsv_bins/glcm_distances/glcm_angles: HSV/GLCM 参数
- smote_k_neighbors/enn_k_neighbors/imbalance_strategy: 类不平衡策略参数
- model_name/svm_c/svm_gamma/svm_probability: SVM 参数
- ann_hidden_layer_sizes/ann_alpha/ann_max_iter: ANN 参数
- split_mode/per_class_limit/train_per_class/val_per_class: 数据划分与每类样本数

- coverage_enabled/coverage_max_rounds: coverage 多轮评估设置（会重复并训练多轮，避免偶然性）
- two_stage_enabled/stage1_imbalance_strategy/stage2_imbalance_strategy: 两阶段训练设置（先粗分后细分，不必太关注，因为没有实质性改善，一般把nv mel和 其他先区分开再区分少数类，但事实证明这个设想难以实现）

- threshold_tuning_enabled/threshold_search_min/threshold_search_max/threshold_search_steps: 阈值搜索配置(这是一个针对少数类进行优化的策略，机器学习中会用到)
- minority_recall_floor/minority_classes: 少数类召回约束与类列表，直白地说就是哪些算是少数类是由这个参数定义的

```

### 运行实例
artifacts_ann_all_features：ANN + 全特征方案，固定抽取均衡样本；
artifacts_ann_full_dataset： 同上，全数据集分类、验证；
artifacts_ann_paper_split: ANN，只使用LBP(BIMF1)+GCLM，固定抽取均衡样本；
artifacts_ann_smote_texture：ANN，特征同上，全数据集，smote策略；
artifacts_ann_texture_only：ANN，特征同上，全数据集，但是没有使用不均衡策略；
artifacts_best_practice：下有ABCDEF 6个实验，主要是进行两步法测试，SVM全特征；A是对照组，BCDE都是两步分类，包括不均等策略调整和阈值搜索的一些参数阈值；这一部分比较复杂，有必要再补充（毕竟指标也没有很明显的提升）；
artifacts_coverage：ANN 全特征，固定抽取均衡样本，反复多次测试，（59次），主要目的是复现所谓文献中训练子集的方法，结果不堪入目；
artifacts_full：SVM带概率校正，全数据集全特征；
artifacts_paper：ANN + LBP（BIMF1）only，是对提出这一方法文献的针对性复现尝试，但由于固定抽取均衡样本，结果不堪入目；
artifacts_robust_svm：去除了HSV特征，适当提高svm_c的值；使用SVM分类
artifacts_smoteenn_ann：ANN全特征，全样本分类，主要是引入了Smote-enn来应对样本不均衡性，但也没有明显改进；
artifacts_svm_smote_hsv_nopre：全特征全样本SVM+Smote，跳过了预处理，用的raw data做分类的。吊诡的是，准确率反而高了——但是严重坍缩至nv，其实也算不上理想。




