# -*- coding: utf-8 -*-
"""
Configuration Module | 配置模块

System configuration and constants for composite material defect detection.
复合材料缺陷检测系统的配置参数和常量定义。

This module centralizes all configuration parameters including model hyperparameters,
training settings, and output paths.
本模块集中管理所有配置参数，包括模型超参数、训练设置和输出路径。
"""

import warnings
import matplotlib.pyplot as plt


# Suppress warnings | 忽略警告信息
warnings.filterwarnings('ignore')


# =============================================================================
# Plotting Configuration | 绘图配置
# =============================================================================

PLOT_STYLE = 'seaborn-v0_8-whitegrid'
FIGURE_SIZE = (12, 8)
FONT_SIZE = 11

# Apply matplotlib settings | 应用matplotlib配置
plt.style.use(PLOT_STYLE)
plt.rcParams['figure.figsize'] = FIGURE_SIZE
plt.rcParams['font.size'] = FONT_SIZE


# =============================================================================
# Diffusion Model Hyperparameters | 扩散模型超参数
# =============================================================================

DIFFUSION_CONFIG = {
    'timesteps': 1000,          # Total diffusion steps | 总扩散步数
    'beta_start': 0.0001,       # Initial noise schedule | 初始噪声调度
    'beta_end': 0.02            # Final noise schedule | 最终噪声调度
}


# =============================================================================
# Zero-shot Learning Hyperparameters | 零样本学习超参数
# =============================================================================

ZERO_SHOT_CONFIG = {
    'visual_dim': 64,           # Visual feature dimension | 视觉特征维度
    'semantic_dim': 32,         # Semantic feature dimension | 语义特征维度
    'embedding_dim': 128,       # Joint embedding dimension | 联合嵌入维度
    'n_heads': 8                # Attention heads count | 注意力头数量
}


# =============================================================================
# Training Configuration | 训练配置
# =============================================================================

TRAINING_CONFIG = {
    'n_epochs': 100,                # Training epochs | 训练轮数
    'n_samples_per_class': 200,     # Samples per class | 每类样本数
    'test_size': 0.2,               # Test set ratio | 测试集比例
    'random_state': 42              # Random seed | 随机种子
}


# =============================================================================
# Loss Function Weights | 损失函数权重
# =============================================================================

LOSS_WEIGHTS = {
    'lambda1': 1.0,     # Diffusion loss weight | 扩散损失权重
    'lambda2': 1.5,     # Classification loss weight | 分类损失权重
    'lambda3': 0.3,     # MMD loss weight | MMD损失权重
    'lambda4': 0.01     # Regularization loss weight | 正则化损失权重
}


# =============================================================================
# Classifier Configuration | 分类器配置
# =============================================================================

CLASSIFIER_CONFIGS = {
    'Neural Network': {
        'hidden_layer_sizes': (128, 64, 32),
        'activation': 'relu',
        'max_iter': 300,
        'early_stopping': True,
        'random_state': 42
    },
    'Random Forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'random_state': 42
    },
    'Gradient Boosting': {
        'n_estimators': 100,
        'max_depth': 6,
        'random_state': 42
    }
}


# =============================================================================
# Defect Type Definitions | 缺陷类型定义
# =============================================================================

# All supported defect types | 所有支持的缺陷类型
DEFECT_TYPES = [
    'Crack',
    'Scratch',
    'Inclusion',
    'Porosity',
    'Delamination',
    'Fiber_Break'
]

# Known defect types for training | 训练集中的已知缺陷类型
SEEN_DEFECTS = [
    'Crack',
    'Scratch',
    'Inclusion',
    'Porosity'
]

# Unknown defect types for zero-shot testing | 零样本测试的未知缺陷类型
UNSEEN_DEFECTS = [
    'Delamination',
    'Fiber_Break'
]


# =============================================================================
# Dimensionality Reduction Configuration | 降维配置
# =============================================================================

TSNE_CONFIG = {
    'n_components': 2,      # Output dimensions | 输出维度
    'random_state': 42,     # Random seed | 随机种子
    'perplexity': 30        # Perplexity parameter | 困惑度参数
}


# =============================================================================
# Output Configuration | 输出配置
# =============================================================================

OUTPUT_DIR = '/home/claude/'

OUTPUT_FILES = {
    'training_curves': 'defect_training_curves.png',
    'model_comparison': 'defect_model_comparison.png',
    'zero_shot_comparison': 'defect_zero_shot_comparison.png',
    'tsne_visualization': 'defect_tsne_visualization.png',
    'ablation_study': 'defect_ablation_study.png',
    'confusion_matrix': 'defect_confusion_matrix.png',
    'generative_comparison': 'defect_generative_comparison.png'
}


# =============================================================================
# Color Scheme | 色彩方案
# =============================================================================

COLOR_PALETTE = {
    'primary': '#3498db',       # Blue | 蓝色
    'secondary': '#e74c3c',     # Red | 红色
    'success': '#2ecc71',       # Green | 绿色
    'neutral': '#95a5a6',       # Gray | 灰色
    'warning': '#f39c12',       # Orange | 橙色
    'info': '#9b59b6'           # Purple | 紫色
}


# =============================================================================
# System Constants | 系统常量
# =============================================================================

# Feature dimension for defect samples | 缺陷样本特征维度
FEATURE_DIM = 64

# Minimum samples required for training | 训练所需最小样本数
MIN_SAMPLES = 10

# Maximum number of classes | 最大类别数
MAX_CLASSES = 10

# Model save format | 模型保存格式
MODEL_FORMAT = 'pkl'

# Figure DPI for saved images | 保存图像的DPI
FIGURE_DPI = 150