"""
音素分类（Phoneme Classification）深度学习项目
==============================================

【项目背景】
音素是语音识别的基本单位，就像文字的字母一样。这个项目的目标是：
- 输入：音频的MFCC特征（39维向量）
- 输出：预测该音频帧属于哪个音素（41个音素类别之一）

【为什么重要】
音素分类是语音识别系统的核心组件，准确的音素识别是语音转文字的基础。

【技术亮点】
1. 使用BiLSTM捕获音频的时序特性
2. 序列级处理，考虑上下文信息
3. 完善的防过拟合策略
4. 高效的变长序列处理

作者：深度学习初学者友好版
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import random
import os
from tqdm import tqdm


def same_seeds(seed):
    """
    设置随机种子，确保实验结果可重复

    【为什么需要】
    深度学习中有很多随机过程（权重初始化、数据shuffle、dropout等），
    设置固定种子可以让每次运行得到相同结果，便于调试和比较。

    【参数】
    seed (int): 随机种子数值

    【涉及的随机性】
    - NumPy随机数生成
    - Python内置随机数
    - PyTorch CPU随机数
    - PyTorch GPU随机数
    - CUDNN后端的随机性
    """
    random.seed(seed)  # Python内置random模块
    np.random.seed(seed)  # NumPy随机数
    torch.manual_seed(seed)  # PyTorch CPU随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 当前GPU随机数
        torch.cuda.manual_seed_all(seed)  # 所有GPU随机数
    # 关闭CUDNN的优化，确保结果完全一致（但会稍微慢一些）
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_feat(path):
    """
    加载预处理的MFCC特征文件

    【MFCC特征说明】
    MFCC (Mel-Frequency Cepstral Coefficients) 是音频信号处理中的经典特征：
    - 模拟人耳对声音的感知特性
    - 维度通常是39维（13个静态 + 13个一阶差分 + 13个二阶差分）
    - 比原始音频波形更适合机器学习模型处理

    【参数】
    path (str): 特征文件路径，通常是.pt格式（PyTorch张量）

    【返回】
    torch.Tensor: 加载的特征张量
    """
    feat = torch.load(path)
    return feat


# 以下被注释的代码是传统的特征拼接方法，现在我们用更先进的序列模型替代
# def shift(x, n):
#     """
#     时间帧移位辅助函数
#
#     【传统方法】
#     在RNN普及之前，常用的方法是手动拼接相邻帧的特征，
#     为模型提供时间上下文信息。例如：
#     - 当前帧 + 前2帧 + 后2帧 = 5帧拼接
#     - 特征维度从39维变成195维 (39 × 5)
#
#     【为什么现在不用】
#     BiLSTM能自动学习时序依赖关系，比手工拼接更强大
#     """


# def concat_feat(x, concat_n):
#     """
#     连接相邻帧特征的传统方法
#
#     【原理】
#     音素识别需要考虑上下文，因为：
#     - 同一个音素在不同上下文中声学特性会变化
#     - 相邻音素之间存在协调发音效应
#
#     【现代替代方案】
#     RNN/LSTM/Transformer等序列模型能更好地建模时序关系
#     """


# class LibriDataset(Dataset):
#     """
#     传统的数据集实现（逐帧处理）
#
#     【问题】
#     - 将序列打散成单独的帧，丢失了序列结构
#     - 无法充分利用RNN的序列建模能力
#     - 内存使用效率低
#     """


class LibriDataset(Dataset):
    """
    改进的序列级数据集类

    【设计理念】
    以完整的句子（utterance）为单位进行处理，这样：
    1. 保持了音频的自然序列结构
    2. 让RNN模型能够充分发挥序列建模能力
    3. 内存使用更高效（按需加载）
    4. 更符合实际应用场景

    【数据组织】
    - 训练集：用于学习模型参数
    - 验证集：用于调整超参数和早停
    - 测试集：用于最终性能评估
    """

    def __init__(self, split, feat_dir, phone_path, train_ratio=0.8):
        """
        初始化数据集

        【参数说明】
        split (str): 数据集划分 - "train"（训练）, "val"（验证）, "test"（测试）
        feat_dir (str): 特征文件目录路径
        phone_path (str): 音素标签文件目录路径
        train_ratio (float): 训练集占比，剩余部分作为验证集

        【数据集划分原理】
        - 训练集：用于梯度下降优化模型参数
        - 验证集：用于超参数调优和防止过拟合
        - 测试集：最终性能评估，不能用于调参
        """
        self.split = split
        self.feat_dir = feat_dir

        # 根据split确定工作模式和文件路径
        if split == "train" or split == "val":
            mode = "train"  # 训练和验证都使用训练数据的文件
            usage_list_path = os.path.join(phone_path, "train_split.txt")
            label_path = os.path.join(phone_path, f"{mode}_labels.txt")
        else:  # split == "test"
            mode = "test"
            usage_list_path = os.path.join(phone_path, "test_split.txt")
            label_path = None  # 测试集没有标签（这是我们要预测的）

        # 加载文件名列表
        with open(usage_list_path) as f:
            usage_list = f.read().splitlines()

        # 如果是训练或验证模式，需要划分数据集
        if split == "train" or split == "val":
            train_len = int(len(usage_list) * train_ratio)
            if split == "train":
                self.usage_list = usage_list[:train_len]  # 前80%作为训练集
            else:  # split == "val"
                self.usage_list = usage_list[train_len:]  # 后20%作为验证集
        else:  # split == "test"
            self.usage_list = usage_list

        print(f"[Dataset] - {split}集句子数量: {len(self.usage_list)}")

        # 加载标签字典（仅在有标签的情况下）
        self.label_dict = {}
        if label_path:
            with open(label_path) as f:
                for line in f.read().splitlines():
                    parts = line.strip().split()
                    # parts[0]是文件名，parts[1:]是该句子每一帧的音素标签
                    self.label_dict[parts[0]] = [int(p) for p in parts[1:]]

    def __len__(self):
        """返回数据集大小（句子数量）"""
        return len(self.usage_list)

    def __getitem__(self, idx):
        """
        获取单个数据样本

        【返回格式】
        - 训练/验证模式：(features, labels)
          - features: (seq_len, 39) 一个句子的MFCC特征序列
          - labels: (seq_len,) 对应每一帧的音素标签
        - 测试模式：features
          - features: (seq_len, 39) 待预测的特征序列

        【为什么返回整个序列】
        RNN模型需要完整的序列信息来建模时序依赖关系
        """
        fname = self.usage_list[idx]
        # 构造特征文件路径
        feat_path = os.path.join(self.feat_dir, "train" if self.split != "test" else "test", f"{fname}.pt")

        # 加载单个句子的MFCC特征
        features = torch.load(feat_path)  # Shape: (seq_len, 39)

        if self.split == "test":
            return features  # 测试集只返回特征，没有标签
        else:
            # 训练和验证集需要返回特征和对应的标签
            labels = torch.LongTensor(self.label_dict[fname])  # Shape: (seq_len,)
            return features, labels


def collate_fn(batch):
    """
    自定义的批处理函数，处理变长序列

    【问题背景】
    不同句子的长度不同，但PyTorch的DataLoader需要将多个样本组成固定大小的batch。
    这就需要对短序列进行填充（padding），使所有序列长度一致。

    【解决方案】
    1. 找到batch中最长的序列
    2. 用特殊值填充短序列到相同长度
    3. 记录每个序列的原始长度，避免模型处理填充部分

    【参数】
    batch (list): DataLoader传入的一个batch的数据

    【返回】
    根据模式不同：
    - 训练/验证: (padded_features, padded_labels, lengths)
    - 测试: (padded_features, lengths)
    """
    # 判断是训练/验证模式还是测试模式
    if isinstance(batch[0], tuple):
        # 训练和验证模式：每个元素是(features, labels)元组
        features, labels = zip(*batch)

        # 记录每个序列的原始长度，这很重要！
        lengths = torch.LongTensor([len(f) for f in features])

        # 使用PyTorch提供的pad_sequence函数进行填充
        # batch_first=True 表示输出维度是 (batch_size, seq_len, feature_dim)
        padded_features = pad_sequence(features, batch_first=True)

        # 标签填充使用-1，这样在计算损失时可以忽略填充位置
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)

        return padded_features, padded_labels, lengths
    else:
        # 测试模式：每个元素只有features
        features = batch
        lengths = torch.LongTensor([len(f) for f in features])
        padded_features = pad_sequence(features, batch_first=True)
        return padded_features, lengths


class SequentialBossModel(nn.Module):
    """
    基于BiLSTM的序列音素分类模型

    【模型设计思想】
    1. 预处理网络：将原始特征映射到合适的表示空间
    2. BiLSTM网络：捕获双向的时序依赖关系
    3. 分类器：将序列表示映射到音素类别

    【为什么叫"Boss"模型】
    设计目标是超越传统的简单基准模型（Boss Baseline），
    通过序列建模技术显著提升音素分类性能。

    【BiLSTM的优势】
    - 双向处理：同时考虑过去和未来的信息
    - 长期记忆：能够捕获长距离的时序依赖
    - 自动特征学习：不需要手工设计时序特征
    """

    def __init__(self, input_dim=39, output_dim=41, hidden_dim=256, num_layers=3, dropout=0.3):
        """
        初始化模型结构

        【参数说明】
        input_dim (int): 输入特征维度，MFCC是39维
        output_dim (int): 输出类别数，英语音素有41个
        hidden_dim (int): LSTM隐藏层维度，控制模型容量
        num_layers (int): LSTM层数，更深的网络学习能力更强
        dropout (float): Dropout比例，防止过拟合

        【网络结构设计原则】
        - 预处理网络：提升特征表示质量
        - 多层BiLSTM：逐层抽象时序模式
        - 适当的正则化：保证泛化能力
        """
        super().__init__()

        # 预处理网络：将39维MFCC特征映射到模型的隐藏维度
        # 为什么需要预处理：
        # 1. 特征归一化和非线性变换
        # 2. 维度适配，为后续LSTM准备
        # 3. 提供初步的特征抽象
        self.prenet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 线性变换
            nn.LayerNorm(hidden_dim),  # 层归一化，稳定训练
            nn.ReLU(),  # 非线性激活函数
            nn.Dropout(dropout),  # 随机失活，防止过拟合
        )

        # BiLSTM网络：核心的序列建模组件
        # 为什么选择BiLSTM：
        # 1. 双向：音素识别需要考虑前后文
        # 2. LSTM：比普通RNN更好地处理长序列
        # 3. 多层：提供分层的特征抽象
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,  # 输入维度
            hidden_size=hidden_dim,  # 隐藏状态维度
            num_layers=num_layers,  # 层数
            bidirectional=True,  # 双向处理
            batch_first=True,  # 输入格式 (batch, seq, feature)
            dropout=dropout if num_layers > 1 else 0,  # 层间dropout
        )

        # 分类器：将BiLSTM的输出映射到音素类别
        # 注意：BiLSTM输出维度是 hidden_dim * 2（双向拼接）
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        """
        前向传播过程

        【参数】
        x: (batch_size, max_seq_len, feature_dim) 填充后的特征序列
        lengths: (batch_size,) 每个序列的原始长度

        【返回】
        logits: (batch_size, max_seq_len, output_dim) 每帧的类别预测分数

        【处理流程详解】
        """
        # 第一步：预处理网络
        # 将39维MFCC特征映射到hidden_dim维空间
        x = self.prenet(x)  # -> (batch_size, max_seq_len, hidden_dim)

        # 第二步：打包变长序列
        # 这是处理填充序列的标准做法，可以：
        # 1. 避免模型处理填充部分
        # 2. 提高计算效率
        # 3. 得到正确的隐藏状态
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # 第三步：通过BiLSTM处理序列
        # BiLSTM会自动学习时序依赖关系
        packed_out, _ = self.bilstm(packed_x)

        # 第四步：解包序列，还原为填充格式
        # 这样后续处理更方便
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # lstm_out shape: (batch_size, max_seq_len, hidden_dim * 2)

        # 第五步：分类器进行逐帧预测
        # 每一个时间步都要预测对应的音素类别
        logits = self.classifier(lstm_out)
        # logits shape: (batch_size, max_seq_len, output_dim=41)

        return logits


class AntiOverfittingTrainer:
    """
    专门设计用于对抗过拟合的训练器

    【过拟合问题】
    深度学习模型容易在训练集上表现很好，但在新数据上表现差。
    这种现象叫过拟合，是深度学习的核心挑战之一。

    【对抗策略】
    1. 正则化技术：权重衰减、Dropout、标签平滑
    2. 学习率调度：动态调整学习率
    3. 早停机制：防止过度训练
    4. 梯度裁剪：稳定训练过程

    【训练监控】
    同时跟踪训练集和验证集的性能，及时发现过拟合
    """

    def __init__(self, model, device, num_epochs=50, patience=10):
        """
        初始化训练器

        【参数】
        model: 要训练的神经网络模型
        device: 计算设备（CPU或GPU）
        num_epochs: 最大训练轮数
        patience: 早停耐心值，验证性能多少轮不提升就停止训练
        """
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.best_val_acc = 0.0  # 记录最佳验证准确率
        self.patience = patience
        self.patience_counter = 0  # 早停计数器

        # 训练历史记录，用于分析训练过程
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def create_optimizer_and_scheduler(self, learning_rate=1e-4, weight_decay=1e-5):
        """
        创建优化器和学习率调度器

        【优化器选择：AdamW】
        - Adam：自适应学习率，训练稳定
        - W（权重衰减）：L2正则化，防止过拟合

        【学习率调度：ReduceLROnPlateau】
        当验证准确率停止提升时，自动降低学习率
        这是一种常用的学习率调度策略

        【损失函数：CrossEntropyLoss】
        - 标准的多分类损失函数
        - 标签平滑：减少过拟合，提高泛化能力
        - ignore_index=-1：忽略填充位置的损失计算
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,  # L2正则化强度
        )

        # 基于验证准确率的学习率调度
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            "max",  # 监控验证准确率（越大越好）
            factor=0.5,  # 学习率衰减因子
            patience=3,  # 3轮不提升就降低学习率
        )

        # 交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1, ignore_index=-1  # 标签平滑，防止过拟合  # 忽略填充位置（标签为-1）
        )

    def train_epoch(self, train_loader):
        """
        训练一个epoch（一轮完整的训练数据遍历）

        【训练步骤】
        1. 设置模型为训练模式
        2. 遍历所有批次数据
        3. 前向传播计算损失
        4. 反向传播更新参数
        5. 记录训练指标

        【返回】
        (平均损失, 平均准确率)
        """
        self.model.train()  # 设置为训练模式，启用Dropout和BatchNorm
        total_loss, total_correct, total_frames = 0.0, 0, 0

        # 使用tqdm显示训练进度条
        for features, labels, lengths in tqdm(train_loader, desc="训练中"):
            # 将数据移到GPU（如果可用）
            features, labels = features.to(self.device), labels.to(self.device)

            # 清零梯度（PyTorch需要手动清零）
            self.optimizer.zero_grad()

            # 前向传播：计算模型输出
            outputs = self.model(features, lengths)

            # 计算损失
            # 需要reshape：outputs (B,T,C) -> (B*T,C), labels (B,T) -> (B*T)
            # 这是因为CrossEntropyLoss期望2D输入
            loss = self.criterion(outputs.view(-1, 41), labels.view(-1))

            # 反向传播：计算梯度
            loss.backward()

            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 计算准确率（只考虑非填充部分）
            predicted = torch.argmax(outputs, dim=-1)  # 获取预测类别
            correct_predictions = ((predicted == labels) & (labels != -1)).sum().item()
            num_frames = (labels != -1).sum().item()  # 非填充帧数

            # 累积统计信息
            total_loss += loss.item()
            total_correct += correct_predictions
            total_frames += num_frames

        # 返回平均损失和准确率
        return total_loss / len(train_loader), total_correct / total_frames

    def validate_epoch(self, val_loader):
        """
        验证一个epoch

        【验证特点】
        1. 不需要计算梯度（节省内存和计算）
        2. 模型设置为评估模式（关闭Dropout）
        3. 只计算损失和准确率，不更新参数

        【作用】
        - 监控模型在未见数据上的性能
        - 早停和学习率调度的依据
        - 防止过拟合的重要手段
        """
        self.model.eval()  # 设置为评估模式
        total_loss, total_correct, total_frames = 0.0, 0, 0

        # 禁用梯度计算，节省内存和计算
        with torch.no_grad():
            for features, labels, lengths in tqdm(val_loader, desc="验证中"):
                features, labels = features.to(self.device), labels.to(self.device)

                outputs = self.model(features, lengths)
                loss = self.criterion(outputs.view(-1, 41), labels.view(-1))

                predicted = torch.argmax(outputs, dim=-1)
                correct_predictions = ((predicted == labels) & (labels != -1)).sum().item()
                num_frames = (labels != -1).sum().item()

                total_loss += loss.item()
                total_correct += correct_predictions
                total_frames += num_frames

        return total_loss / len(val_loader), total_correct / total_frames

    def train(self, train_loader, val_loader, model_path="./HW2/bilstm_model.ckpt"):
        """
        完整的训练流程

        【训练循环】
        1. 每个epoch训练一轮
        2. 验证模型性能
        3. 更新学习率
        4. 保存最佳模型
        5. 检查早停条件

        【早停机制】
        如果验证准确率在patience轮内没有提升，就停止训练。
        这防止了过拟合，也节省了训练时间。
        """
        print("🚀 开始序列模型训练...")
        print(f"📊 模型参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("🎯 目标：通过序列建模突破Boss Baseline")
        print("=" * 80)

        for epoch in range(self.num_epochs):
            # 训练一个epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            # 验证一个epoch
            val_loss, val_acc = self.validate_epoch(val_loader)

            # 学习率调度（基于验证准确率）
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 记录训练历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # 计算过拟合差距
            overfitting_gap = train_acc - val_acc

            # 打印训练信息
            print(f"📊 Epoch [{epoch+1:03d}/{self.num_epochs:03d}]")
            print(f"   学习率: {current_lr:.6f}")
            print(f"   训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
            print(f"   验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
            print(f"   🔍 过拟合差距: {overfitting_gap:.4f}")

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), model_path)
                print(f"   💾 保存最佳模型！验证准确率: {self.best_val_acc:.4f}")
                self.patience_counter = 0  # 重置早停计数器
            else:
                self.patience_counter += 1
                print(f"   ⏳ 等待改善... ({self.patience_counter}/{self.patience})")

            print("-" * 80)

            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"🛑 早停触发！在第 {epoch+1} 轮停止训练")
                break

        print("🏁 训练完成！")
        print(f"🏆 最佳验证准确率: {self.best_val_acc:.4f}")
        return self.best_val_acc


def main_improved_training(config, model_path):
    """
    序列模型的主训练流程

    【流程概述】
    1. 环境设置：随机种子、设备选择
    2. 数据加载：训练集、验证集
    3. 模型创建：BiLSTM网络
    4. 训练执行：反过拟合训练器

    【参数】
    config (dict): 超参数配置字典
    model_path (str): 模型保存路径
    """
    # 设置随机种子，确保结果可重复
    same_seeds(config["seed"])

    # 选择计算设备：GPU优先，CPU备选
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  设备: {device}")

    # 定义数据路径（请根据实际情况修改）
    feat_dir = r"E:\Document\code\LHY2023\HW2\libriphone\feat"
    phone_path = r"E:\Document\code\LHY2023\HW2\libriphone"

    # 创建数据集实例
    print("\n📂 加载数据集...")
    train_set = LibriDataset("train", feat_dir, phone_path, config["train_ratio"])
    val_set = LibriDataset("val", feat_dir, phone_path, config["train_ratio"])

    # 创建数据加载器
    # DataLoader负责批处理、打乱数据、多进程加载等
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],  # 批大小
        shuffle=True,  # 打乱数据顺序
        num_workers=0,  # 数据加载进程数
        collate_fn=collate_fn,  # 自定义批处理函数
        pin_memory=True,  # 加速GPU传输
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,  # 验证时不需要打乱
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 创建序列模型
    model = SequentialBossModel(
        input_dim=39,  # MFCC特征维度
        output_dim=41,  # 音素类别数
        hidden_dim=config["hidden_dim"],  # 隐藏层维度
        num_layers=config["num_layers"],  # LSTM层数
        dropout=config["dropout_rate"],  # Dropout比例
    ).to(
        device
    )  # 移动到计算设备

    # 创建并配置训练器
    trainer = AntiOverfittingTrainer(model, device, num_epochs=config["num_epochs"], patience=config["patience"])
    trainer.create_optimizer_and_scheduler(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])

    # 开始训练
    best_acc = trainer.train(train_loader, val_loader, model_path=model_path)

    print("\n🎊 序列模型训练完成！")


def test_and_predict(model_path, config):
    """
    使用训练好的模型进行测试和预测

    【预测流程】
    1. 加载测试数据
    2. 重建训练好的模型
    3. 逐批预测音素类别
    4. 保存预测结果到CSV文件

    【注意事项】
    - 模型结构必须与训练时完全一致
    - 需要正确处理变长序列
    - 预测结果格式要符合提交要求

    【参数】
    model_path (str): 训练好的模型文件路径
    config (dict): 模型配置，确保结构一致
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔮 开始测试预测，设备: {device}")

    # 数据路径（与训练时相同）
    feat_dir = r"E:\Document\code\LHY2023\HW2\libriphone\feat"
    phone_path = r"E:\Document\code\LHY2023\HW2\libriphone"

    # 加载测试数据集
    print("📂 加载测试数据...")
    test_set = LibriDataset("test", feat_dir, phone_path)
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn  # 测试时不打乱顺序
    )
    print(f"✅ 测试数据加载完成！测试集句子数: {len(test_set):,}")

    # 重建模型结构（必须与训练时完全一致）
    model = SequentialBossModel(
        input_dim=39,
        output_dim=41,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout_rate"],
    ).to(device)

    # 加载训练好的模型参数
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"❌ 错误: 找不到模型文件 '{model_path}'。请确认路径是否正确。")
        return None

    model.eval()  # 设置为评估模式
    print(f"📁 已加载最佳模型: {model_path}")

    # 开始预测
    print("🔮 正在生成预测结果...")
    all_predictions = []

    with torch.no_grad():  # 禁用梯度计算
        for features, lengths in tqdm(test_loader, desc="预测中"):
            features = features.to(device)

            # 模型前向传播
            outputs = model(features, lengths)

            # 获取预测类别（概率最大的类别）
            predicted_labels = torch.argmax(outputs, dim=-1)

            # 提取每个序列的有效预测（排除填充部分）
            for i in range(len(lengths)):
                valid_length = lengths[i]
                all_predictions.extend(predicted_labels[i, :valid_length].cpu().numpy())

    # 转换为numpy数组
    final_predictions = np.array(all_predictions, dtype=np.int32)
    print("✅ 预测完成！")
    print(f"📊 共预测 {len(final_predictions):,} 个音素帧")

    # 保存预测结果为CSV格式
    output_file = r"E:\Document\code\LHY2023\HW2\prediction_bilstm.csv"
    with open(output_file, "w") as f:
        f.write("Id,Class\n")  # CSV头部
        for i, y in enumerate(final_predictions):
            f.write(f"{i},{y}\n")

    print(f"💾 预测结果已保存到: {output_file}")
    return final_predictions


if __name__ == "__main__":
    """
    主程序入口

    【超参数配置说明】
    - train_ratio: 训练集比例，影响模型的学习数据量
    - batch_size: 批大小，影响训练稳定性和内存使用
    - num_epochs: 最大训练轮数
    - patience: 早停耐心值，防止过拟合
    - learning_rate: 学习率，控制参数更新步长
    - weight_decay: 权重衰减，L2正则化强度
    - seed: 随机种子，确保结果可重复
    - hidden_dim: 隐藏层维度，控制模型容量
    - num_layers: LSTM层数，影响模型复杂度
    - dropout_rate: Dropout比例，防止过拟合

    【超参数调优建议】
    1. 先用小模型快速验证流程
    2. 逐步增加模型复杂度
    3. 观察训练/验证曲线，调整正则化强度
    4. 使用验证集性能指导超参数选择
    """

    # 超参数配置字典
    config = {
        "train_ratio": 0.8,  # 80%数据用于训练，20%用于验证
        "batch_size": 32,  # 批大小，平衡内存使用和训练稳定性
        "num_epochs": 100,  # 最大训练轮数
        "patience": 20,  # 早停耐心值，防止过拟合
        "learning_rate": 5e-4,  # 学习率，控制参数更新步长
        "weight_decay": 1e-5,  # 权重衰减，L2正则化
        "seed": 3407,  # 随机种子，确保结果可重复
        "hidden_dim": 512,  # LSTM隐藏维度，控制模型容量
        "num_layers": 4,  # LSTM层数，更深的网络学习能力更强
        "dropout_rate": 0.4,  # Dropout比例，防止过拟合
    }

    # 统一的模型保存路径
    MODEL_SAVE_PATH = "./bilstm_model_final.ckpt"

    print("🎯 高级音素分类：序列模型解决方案")
    print("=" * 60)

    try:
        # 第一阶段：模型训练
        print("🚀 第一阶段：开始训练序列模型...")
        main_improved_training(config, MODEL_SAVE_PATH)

        # 第二阶段：测试预测
        print("\n🔮 第二阶段：开始测试预测...")
        predictions = test_and_predict(MODEL_SAVE_PATH, config)
        if predictions is not None:
            print("\n🎉 全部完成！请提交 prediction_bilstm.csv 文件")
            print("\n📈 性能提升建议：")
            print("1. 尝试不同的网络结构（层数、维度）")
            print("2. 调整正则化强度（dropout、weight_decay）")
            print("3. 使用学习率调度策略")
            print("4. 考虑数据增强技术")
            print("5. 尝试其他序列模型（Transformer）")

    except Exception as e:
        import traceback

        print(f"❌ 主流程出现错误: {e}")
        traceback.print_exc()
        print("请检查数据路径和配置")
        print("\n🔧 常见问题排查：")
        print("1. 检查数据文件路径是否正确")
        print("2. 确认GPU内存是否足够")
        print("3. 验证Python包版本兼容性")
        print("4. 查看详细错误信息进行调试")
