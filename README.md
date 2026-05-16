# HAM10000_EVALUATE (ICICT 2024 Reproduction)

本仓库为以下论文的工程化复现实验代码，聚焦传统机器学习路线，并针对 HAM10000 七分类任务实现可复用、可扩展、可缓存的实验流水线：

- Random Forest
- K-Nearest Neighbors (KNN)
- SVM (Polynomial Kernel)
- 数据不平衡处理：SMOTE、SMOTE-ENN

## 1. 复现目标

对论文中强调的方法进行端到端复现：

1. 基于 HAM10000 元数据和图像构建监督学习数据集
2. 提取可复现实用的手工统计特征（颜色 + 纹理）
3. 对比三种采样策略：`none` / `smote` / `smoteenn`
4. 对比三类模型：`random_forest` / `knn` / `svm_poly`
5. 统一导出 accuracy、precision、recall、F1、混淆矩阵与排行榜

## 2. 目录结构

```text
.
├── main.py
├── config.py
├── requirements.txt
├── README.md
└── reproducer/
    ├── __init__.py
    ├── cache.py
    ├── data.py
    ├── experiment.py
    ├── features.py
    ├── metrics.py
    ├── models.py
    └── sampling.py
```

## 3. 环境安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 数据准备

确保 `--dataset-root` 目录下至少包含：

```text
HAM10000_metadata.csv
HAM10000_images_part_1/
HAM10000_images_part_2/
```

程序会按 `image_id.jpg` 自动在以上目录中查找图像。

## 5. 运行方式

```bash
python main.py \
  --dataset-root /path/to/HAM10000 \
  --output-dir artifacts \
  --image-size 96 \
  --test-size 0.2 \
  --random-state 42 \
  --n-jobs -1
```

## 6. 输出结果

运行后在 `artifacts/` 下生成：

1. `class_distribution_original.csv`
2. `class_distribution_none.csv` / `class_distribution_smote.csv` / `class_distribution_smoteenn.csv`
3. `leaderboard.csv`
4. 每组实验目录：`artifacts/<sampling>/<model>/`

每组实验目录内包含：

1. `metrics.json`
2. `confusion_matrix.png`
3. `model.joblib`

## 7. 性能设计

为延续原仓库对性能优化的思路，本实现保留并强化了两类机制：

1. 并行：`joblib.Parallel` 对图像特征提取并行执行
2. 缓存：
   - 特征缓存（`features_*.npz` / `labels_*.npy`）
   - 采样缓存（`joblib.Memory`）避免重复 SMOTE/SMOTE-ENN 计算

## 8. 注意事项

1. 本仓库为论文复现工程，不包含医疗用途声明下的临床决策能力。
2. 最终指标受随机种子、硬件、图像解码环境和依赖版本影响。
3. 若你变更特征提取逻辑，建议清理 `artifacts/.cache/` 后重跑。

## 9. 参考文献

1. S. S. K., et al., "Analysis of Skin Lesion Classification using Computational Models," 2024 IEEE International Conference on Inventive Computation Technologies (ICICT), 2024.
2. DOI: [10.1109/ICICT60155.2024.10673538](https://doi.org/10.1109/ICICT60155.2024.10673538)

建议引用格式（BibTeX，可按 IEEE Xplore 实际条目补全作者字段）：

```bibtex
@inproceedings{icict2024_skin_lesion_models,
  title={Analysis of Skin Lesion Classification using Computational Models},
  booktitle={2024 IEEE International Conference on Inventive Computation Technologies (ICICT)},
  year={2024},
  doi={10.1109/ICICT60155.2024.10673538}
}
```
