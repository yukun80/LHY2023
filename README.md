# 李宏毅机器学习课程2023 - 作业实现

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 项目简介

本项目是李宏毅教授**机器学习课程2023**的作业实现代码仓库。目标是通过实际编程练习，深入理解机器学习的核心概念和算法实现，涵盖从传统机器学习到深度学习的各个重要主题。

> **课程特色**: 李宏毅教授的课程以其深入浅出的讲解和实战性强的作业而闻名，是华语机器学习教育的经典课程。

## 🎯 学习目标

- 掌握机器学习的数学基础和算法原理
- 熟练使用PyTorch等深度学习框架
- 培养解决实际机器学习问题的能力
- 理解从传统ML到深度学习的技术演进

## 📚 作业完成情况

### ✅ 已完成作业


- [X] **HW1 - 回归预测 (Regression)**

  - 📁 路径: `HW1/`
  - 🎯 任务: COVID-19确诊数预测
  - 🔧 技术: 深度神经网络(DNN) + 特征选择 + 正则化
  - 💡 亮点:
    - 使用RFE递归特征消除进行特征选择
    - Adam优化器 + 学习率衰减 + 权重衰减
    - 早停机制防止过拟合
    - TensorBoard可视化训练过程
  - 📊 模型: 4层全连接神经网络 (64→32→16→1)
  - 📋 文件:
    - `hw1_win10_Local.ipynb` - 本地运行版本
    - `hw1-Kaggle.ipynb` - Kaggle平台版本
    - `covid_train.csv` - 训练数据
    - `covid_test.csv` - 测试数据
    - `models/model.ckpt` - 训练好的模型

- [X] **HW2 - 音素分类 (Phoneme Classification)**

  - 📁 路径: `HW2/`
  - 🎯 任务: 使用MFCC特征进行41类音素分类
  - 🔧 技术: BiLSTM序列模型 + 反过拟合训练策略
  - 💡 亮点:
    - 序列级数据处理，保持时序结构
    - 变长序列的高效批处理
    - 完善的正则化和早停机制
    - 详细的教育性注释，适合初学者理解
  - 📊 模型: SequentialBossModel (BiLSTM架构)
  - 📋 文件:
    - `hw2-boss-code-fyk.py` - 主实现代码
    - `hw2-boss-code-fyk.ipynb` - Jupyter版本
    - `HW2_Guide.md` - 作业指南
    - `README_本地运行指南.md` - 运行说明
- [X] **HW1 - 回归预测 (Regression)**

  - 📁 路径: `HW1/`
  - 🎯 任务: COVID-19确诊数预测
  - 🔧 技术: 深度神经网络(DNN) + 特征选择 + 正则化
  - 💡 亮点:
    - 使用RFE递归特征消除进行特征选择
    - Adam优化器 + 学习率衰减 + 权重衰减
    - 早停机制防止过拟合
    - TensorBoard可视化训练过程
  - 📊 模型: 4层全连接神经网络 (64→32→16→1)
  - 📋 文件:
    - `hw1_win10_Local.ipynb` - 本地运行版本
    - `hw1-Kaggle.ipynb` - Kaggle平台版本
    - `covid_train.csv` - 训练数据
    - `covid_test.csv` - 测试数据
    - `models/model.ckpt` - 训练好的模型

### ⏳ 待完成作业

- [ ] **HW3 - 卷积神经网络 (Convolutional Neural Network)**

  - 🎯 任务: 食物图像分类
  - 🔧 技术: CNN、数据增强、Transfer Learning
- [ ] **HW4 - 自注意力机制 (Self-Attention)**

  - 🎯 任务: 说话人分类
  - 🔧 技术: Transformer、Self-Attention
- [ ] **HW5 - 机器翻译 (Machine Translation)**

  - 🎯 任务: 英翻中机器翻译
  - 🔧 技术: Seq2Seq、Attention机制
- [ ] **HW6 - 生成对抗网络 (Generative Adversarial Network)**

  - 🎯 任务: 动漫头像生成
  - 🔧 技术: GAN、DCGAN、条件生成
- [ ] **HW7 - BERT文本分类 (BERT)**

  - 🎯 任务: 情感分析
  - 🔧 技术: BERT预训练模型、Fine-tuning
- [ ] **HW8 - 异常检测 (Anomaly Detection)**

  - 🎯 任务: 图像异常检测
  - 🔧 技术: AutoEncoder、重构误差
- [ ] **HW9 - 可解释人工智能 (Explainable AI)**

  - 🎯 任务: CNN可视化
  - 🔧 技术: Grad-CAM、Saliency Map
- [ ] **HW10 - 对抗攻击 (Adversarial Attack)**

  - 🎯 任务: 对抗样本生成
  - 🔧 技术: FGSM、PGD攻击
- [ ] **HW11 - 域适应 (Domain Adaptation)**

  - 🎯 任务: 跨域图像分类
  - 🔧 技术: DANN、域对抗训练
- [ ] **HW12 - 强化学习 (Reinforcement Learning)**

  - 🎯 任务: 游戏AI
  - 🔧 技术: DQN、Policy Gradient

## 🛠️ 技术栈

### 核心框架

- **PyTorch** - 深度学习框架
- **NumPy** - 数值计算
- **Pandas** - 数据处理
- **Matplotlib/Seaborn** - 数据可视化

### 开发工具

- **Jupyter Notebook** - 交互式开发
- **tqdm** - 进度条显示
- **Git** - 版本控制

### 硬件要求

- **CPU**: 多核处理器（推荐）
- **GPU**: NVIDIA GPU with CUDA（可选，但强烈推荐）
- **内存**: 8GB+（推荐16GB+）

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/your-username/LHY2023.git
cd LHY2023

# 创建虚拟环境
python -m venv lhy2023
source lhy2023/bin/activate  # Linux/Mac
# 或
lhy2023\Scripts\activate     # Windows

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install jupyter tqdm scikit-learn
```

### 运行示例（以HW2为例）

```bash
cd HW2
python hw2-boss-code-fyk.py
```

## 📊 项目结构

```
LHY2023/
├── README.md                 # 项目说明
├── .gitignore               # Git忽略文件
├── HW1/                     # 回归预测（已完成）
│   ├── hw1_win10_Local.ipynb
│   ├── hw1-Kaggle.ipynb
│   ├── covid_train.csv
│   ├── covid_test.csv
│   └── models/
│       └── model.ckpt
├── HW2/                     # 音素分类（已完成）
│   ├── hw2-boss-code-fyk.py
│   ├── hw2-boss-code-fyk.ipynb
│   ├── HW2_Guide.md
│   └── README_本地运行指南.md
├── HW3/                     # CNN图像分类（待完成）
├── HW4/                     # Self-Attention（待完成）
├── HW5/                     # 机器翻译（待完成）
├── HW6/                     # GAN生成（待完成）
├── HW7/                     # BERT文本分类（待完成）
├── HW8/                     # 异常检测（待完成）
├── HW9/                     # 可解释AI（待完成）
├── HW10/                    # 对抗攻击（待完成）
├── HW11/                    # 域适应（待完成）
└── HW12/                    # 强化学习（待完成）
```

## 🏆 已完成作业亮点

### HW1 - COVID-19病例预测

- **🎯 完整的回归解决方案**: 使用深度神经网络进行COVID-19确诊数预测
- **🔧 特征工程优化**: 采用RFE递归特征消除，从88个特征中选择最重要的20个
- **📊 模型优化**: 4层全连接网络 + Adam优化器 + 学习率衰减
- **⚡ 技术特色**:
  - 多种优化器对比实验 (SGD, Adam, RMSprop)
  - 权重衰减防止过拟合
  - 早停机制自动终止训练
  - TensorBoard可视化训练过程

### HW2 - 音素分类

- **🎯 超越Boss Baseline**: 使用BiLSTM序列建模显著提升性能
- **🔧 工程优化**: 高效的变长序列处理和内存管理
- **📚 教育价值**: 详细注释，适合深度学习初学者学习
- **⚡ 技术特色**:
  - 序列级数据处理
  - 多层双向LSTM
  - 完善的正则化策略
  - 自动早停和学习率调度

## 📈 学习进度跟踪

- **总进度**: 2/12 完成 (16.7%)
- **已完成**: 回归预测 ✅ + 音素分类 ✅
- **当前专注**: 深度学习基础 → 计算机视觉 → 自然语言处理 → 高级主题

## 🔗 相关资源

- [李宏毅机器学习课程官网](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)
- [课程视频 - YouTube](https://www.youtube.com/playlist?list=PLJV_el3uVTsPM2mM-OQzJXziCGJa8nJL8)
- [课程PPT和作业](https://github.com/virginiakm1988/ML2023-Spring)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进代码质量和添加新功能！

### 代码规范

- 遵循PEP8编码规范
- 添加详细的函数和类注释
- 提交前运行代码测试

### 提交格式

```
[作业编号] 简短描述

详细说明变更内容
- 修复的问题
- 新增的功能
- 性能改进等
```

## 📄 许可证

本项目采用 [MIT许可证](LICENSE)，欢迎学习和交流使用。

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：

- **GitHub Issues**: 技术问题和bug报告
- **Email**: your-email@example.com
- **微信**: your-wechat-id

---

> **免责声明**: 本项目仅供学习交流使用，请勿直接用于作业提交。鼓励理解算法原理后独立完成作业。

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**
