# Boss Baseline 本地运行指南

## 📁 目录结构确认
请确保您的HW2目录结构如下：
```
HW2/
├── hw2-boss-code-fyk.ipynb     # Boss改进版代码
├── hw2-sample-code.ipynb       # 原始示例代码
├── libriphone/                 # 数据目录
│   ├── feat/                   # 特征数据
│   │   ├── train/              # 训练特征文件 (.pt files)
│   │   └── test/               # 测试特征文件 (.pt files)
│   ├── train_labels.txt        # 训练标签
│   ├── train_split.txt         # 训练数据分割
│   └── test_split.txt          # 测试数据分割
└── README_本地运行指南.md       # 本文件
```

## 🚀 快速开始

### 1. 环境要求
- Python 3.8+
- PyTorch 1.8+
- CUDA（推荐，用于GPU加速）

### 2. 运行步骤
1. **确认当前目录**：确保您在HW2目录下运行代码
2. **打开notebook**：使用Jupyter Lab/Notebook打开 `hw2-boss-code-fyk.ipynb`
3. **逐个运行代码块**：按顺序执行每个cell

### 3. 已修改的配置

#### 数据路径
- ✅ 已从Kaggle路径改为本地相对路径
- ✅ 训练数据：`./libriphone/feat` 和 `./libriphone`
- ✅ 测试数据：`./libriphone/feat` 和 `./libriphone`

#### 系统优化
- ✅ DataLoader的num_workers设为0（适配Windows环境）
- ✅ 添加了路径验证功能
- ✅ 优化了内存管理

### 4. Boss Baseline 特性

#### 🧠 模型架构
- **深度特征提取**：多层全连接 + BatchNorm + Dropout
- **双向LSTM**：3层，捕获时序上下文信息  
- **自注意力机制**：自动学习重要特征权重
- **深度分类器**：4层分类头

#### 📊 训练配置
- **上下文帧数**：11帧（提供丰富上下文）
- **批次大小**：256（稳定梯度更新）
- **学习率调度**：Cosine Annealing + Warmup
- **正则化**：Dropout(0.3) + Weight Decay + Label Smoothing
- **早停**：Patience=8，防止过拟合

#### 🎯 性能目标
- **Boss Baseline目标**：验证准确率 ≥ 83%
- **预期性能**：通常可达到85%+的准确率

### 5. 运行监控

运行第一个数据加载cell时，您会看到：
```
✅ 特征数据目录验证成功: ./libriphone/feat
✅ 音素数据目录验证成功: ./libriphone
✅ train_labels.txt 存在
✅ train_split.txt 存在  
✅ test_split.txt 存在
```

如果看到❌错误提示，请检查：
1. 当前工作目录是否为HW2
2. libriphone文件夹是否完整下载

### 6. 训练过程

训练过程中您会看到：
```
📊 [001/030] LR: 0.000333
   训练 - 损失: 2.156, 准确率: 0.312
   验证 - 损失: 1.876, 准确率: 0.425
   💾 保存最佳模型！验证准确率: 0.42501
```

当达到Boss Baseline时：
```
🎉 恭喜！达到Boss Baseline: 0.83456 >= 83%
```

### 7. 输出文件

训练完成后会生成：
- `boss_model.ckpt`：最佳模型权重
- `boss_baseline_prediction.csv`：测试集预测结果

### 8. 常见问题

**Q: 内存不足怎么办？**
A: 降低batch_size，比如从256改为128

**Q: 训练太慢怎么办？**  
A: 确保使用GPU，或降低模型复杂度

**Q: 准确率不达标怎么办？**
A: 增加训练轮数，调整学习率，或使用更多数据增强

### 9. 技术支持
如有问题，请检查：
1. 数据完整性
2. 环境依赖
3. 硬件资源（GPU内存、系统内存）

祝您训练顺利！🚀 