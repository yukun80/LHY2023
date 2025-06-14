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
    ## 🚀 快速开始
