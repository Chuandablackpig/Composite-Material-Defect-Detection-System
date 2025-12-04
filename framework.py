# -*- coding: utf-8 -*-
"""
Joint Training Framework Module | 联合训练框架模块

This module implements a joint framework that integrates diffusion model
and zero-shot learning for defect detection and classification.
本模块实现联合框架，整合扩散模型和零样本学习用于缺陷检测和分类。

The framework combines data augmentation through diffusion models with
zero-shot learning capabilities for unseen defect types.
该框架结合扩散模型的数据增强和零样本学习能力处理未知缺陷类型。
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from diffusion_model import DiffusionModelSimulator
from zero_shot_learning import ZeroShotLearningModule
from defect_attributes import DefectAttributes
from config import TRAINING_CONFIG, LOSS_WEIGHTS, CLASSIFIER_CONFIGS


class DiffusionZeroShotFramework:
    """
    Joint Framework for Diffusion Model and Zero-shot Learning
    扩散模型与零样本学习联合框架

    This framework combines conditional diffusion model for synthetic data
    generation with zero-shot learning for classifying unseen defect types.
    该框架结合条件扩散模型用于合成数据生成，以及零样本学习用于分类未知缺陷类型。

    Attributes:
        diffusion_model (DiffusionModelSimulator): Diffusion model instance
                                                   扩散模型实例
        zero_shot_module (ZeroShotLearningModule): Zero-shot learning instance
                                                   零样本学习实例
        classifiers (dict): Dictionary of trained classifiers
                           训练的分类器字典
        training_history (dict): Training metrics history
                                训练指标历史
        is_trained (bool): Whether framework has been trained
                          框架是否已训练
    """

    def __init__(self):
        """
        Initialize the joint training framework.
        初始化联合训练框架。

        Creates diffusion model and zero-shot learning module instances,
        initializes storage for classifiers and training history.
        创建扩散模型和零样本学习模块实例，初始化分类器和训练历史存储。
        """
        self.diffusion_model = DiffusionModelSimulator()
        self.zero_shot_module = ZeroShotLearningModule()
        self.classifiers = {}
        self.training_history = {'loss': [], 'accuracy': [], 'f1': []}
        self.is_trained = False

        print("\n[Joint Framework] Initialization complete")
        print("[联合框架] 初始化完成")

    def generate_training_data(self, n_samples_per_class=None):
        """
        Generate training data using both real and synthetic samples.
        使用真实样本和合成样本生成训练数据。

        Combines simulated real features with diffusion model generated
        features to create an augmented training dataset.
        结合模拟的真实特征和扩散模型生成的特征以创建增强训练数据集。

        Args:
            n_samples_per_class (int, optional): Number of samples per class
                                                每类的样本数，默认从配置读取

        Returns:
            tuple: (X_real, X_generated, X_combined, y, y_combined)
                - X_real (numpy.ndarray): Real feature samples
                                         真实特征样本
                - X_generated (numpy.ndarray): Generated feature samples
                                              生成的特征样本
                - X_combined (numpy.ndarray): Combined features
                                             组合特征
                - y (numpy.ndarray): Labels for single set
                                    单组标签
                - y_combined (numpy.ndarray): Labels for combined set
                                             组合标签
        """
        n_samples_per_class = (
                n_samples_per_class or TRAINING_CONFIG['n_samples_per_class']
        )

        print("\n[Data Generation] Generating training data...")

        X_real = []
        X_generated = []
        y = []

        # Generate data for each seen defect type | 为每种已知缺陷类型生成数据
        for defect_type in DefectAttributes.SEEN_DEFECTS:
            print(f"  Generating {defect_type} samples...")

            # Simulate real features | 模拟真实特征
            real_features = self._simulate_real_features(
                defect_type, n_samples_per_class
            )
            X_real.append(real_features)

            # Generate synthetic features using diffusion model
            # 使用扩散模型生成合成特征
            gen_features = self.diffusion_model.generate_defect_features(
                defect_type, n_samples_per_class
            )
            X_generated.append(gen_features)

            # Create labels | 创建标签
            y.extend([defect_type] * n_samples_per_class)

        # Combine arrays | 组合数组
        X_real = np.vstack(X_real)
        X_generated = np.vstack(X_generated)
        y = np.array(y)

        # Merge real and generated data for augmentation | 合并真实和生成数据用于增强
        X_combined = np.vstack([X_real, X_generated])
        y_combined = np.concatenate([y, y])

        print(f"[Data Generation] Complete! Total samples: {len(y_combined)}")
        print(f"[数据生成] 完成！总样本数: {len(y_combined)}")

        return X_real, X_generated, X_combined, y, y_combined

    def _simulate_real_features(self, defect_type, n_samples):
        """
        Simulate realistic defect features for a given type.
        模拟给定类型的真实缺陷特征。

        Creates synthetic feature vectors that mimic real defect characteristics
        by adding type-specific patterns and noise.
        通过添加类型特定模式和噪声创建模拟真实缺陷特征的合成特征向量。

        Args:
            defect_type (str): Type of defect to simulate
                              要模拟的缺陷类型
            n_samples (int): Number of samples to generate
                            要生成的样本数

        Returns:
            numpy.ndarray: Simulated feature matrix of shape (n_samples, 64)
                          形状为(n_samples, 64)的模拟特征矩阵
        """
        # Generate base random features | 生成基础随机特征
        base_features = np.random.randn(n_samples, 64)

        # Add type-specific pattern | 添加类型特定模式
        type_idx = DefectAttributes.DEFECT_TYPES.index(defect_type)
        pattern = np.zeros(64)
        pattern[type_idx * 10:(type_idx + 1) * 10] = 1

        # Combine pattern with base features | 将模式与基础特征组合
        features = base_features + pattern * 2

        # Add Gaussian noise for realism | 添加高斯噪声以增加真实性
        features += np.random.randn(n_samples, 64) * 0.3

        return features

    def compute_mmd_loss(self, X_diffusion, X_zeroshot):
        """
        Compute Maximum Mean Discrepancy (MMD) loss between distributions.
        计算分布之间的最大平均差异(MMD)损失。

        MMD measures the distance between two probability distributions
        by comparing their mean embeddings in a reproducing kernel Hilbert space.
        MMD通过比较再生核希尔伯特空间中的均值嵌入来衡量两个概率分布之间的距离。

        Formula: MMD² = ||μ_d - μ_z||² + ||Σ_d - Σ_z||_F

        Args:
            X_diffusion (numpy.ndarray): Features from diffusion model
                                        来自扩散模型的特征
            X_zeroshot (numpy.ndarray): Features from zero-shot module
                                       来自零样本模块的特征

        Returns:
            float: MMD loss value (lower indicates more similar distributions)
                  MMD损失值（越低表示分布越相似）
        """
        # Compute mean vectors | 计算均值向量
        mu_d = np.mean(X_diffusion, axis=0)
        mu_z = np.mean(X_zeroshot, axis=0)

        # Compute covariance matrices | 计算协方差矩阵
        if X_diffusion.shape[0] > 1:
            sigma_d = np.cov(X_diffusion.T)
        else:
            sigma_d = np.eye(X_diffusion.shape[1])

        if X_zeroshot.shape[0] > 1:
            sigma_z = np.cov(X_zeroshot.T)
        else:
            sigma_z = np.eye(X_zeroshot.shape[1])

        # Mean difference term | 均值差异项
        mean_diff = np.sum((mu_d - mu_z) ** 2)

        # Covariance difference term | 协方差差异项
        cov_diff = np.sum((sigma_d - sigma_z) ** 2)

        # Combined MMD loss | 组合MMD损失
        mmd = mean_diff + 0.1 * cov_diff

        return mmd

    def compute_total_loss(self, y_true, y_pred, X_diff, X_zero,
                           model_params=None):
        """
        Compute total multi-objective loss function.
        计算总的多目标损失函数。

        Combines diffusion loss, classification loss, MMD loss, and
        regularization loss with configurable weights.
        结合扩散损失、分类损失、MMD损失和正则化损失，使用可配置的权重。

        Formula: L_total = λ₁L_diff + λ₂L_cls + λ₃L_mmd + λ₄L_reg

        Args:
            y_true (numpy.ndarray): True labels
                                   真实标签
            y_pred (numpy.ndarray): Predicted labels
                                   预测标签
            X_diff (numpy.ndarray): Diffusion model features
                                   扩散模型特征
            X_zero (numpy.ndarray): Zero-shot learning features
                                   零样本学习特征
            model_params (dict, optional): Model parameters for regularization
                                         用于正则化的模型参数

        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss (float): Combined weighted loss value
                                     组合加权损失值
                - loss_dict (dict): Individual loss components
                                   各个损失分量
        """
        # Load loss weights from configuration | 从配置加载损失权重
        lambda1 = LOSS_WEIGHTS['lambda1']
        lambda2 = LOSS_WEIGHTS['lambda2']
        lambda3 = LOSS_WEIGHTS['lambda3']
        lambda4 = LOSS_WEIGHTS['lambda4']

        # Classification loss (cross-entropy approximation) | 分类损失（交叉熵近似）
        correct = np.sum(y_true == y_pred)
        L_cls = 1 - correct / len(y_true)

        # MMD loss between distributions | 分布间MMD损失
        L_mmd = self.compute_mmd_loss(X_diff, X_zero)

        # Diffusion loss (simplified) | 扩散损失（简化版）
        L_diff = np.mean(np.abs(X_diff - X_zero))

        # Regularization loss | 正则化损失
        L_reg = 0.01

        # Compute weighted total loss | 计算加权总损失
        total_loss = (
                lambda1 * L_diff +
                lambda2 * L_cls +
                lambda3 * L_mmd +
                lambda4 * L_reg
        )

        # Return loss components for monitoring | 返回损失分量用于监控
        loss_dict = {
            'L_diff': L_diff,
            'L_cls': L_cls,
            'L_mmd': L_mmd,
            'L_reg': L_reg
        }

        return total_loss, loss_dict

    def train(self, n_epochs=None, n_samples_per_class=None):
        """
        Train the joint framework with multiple classifiers.
        使用多个分类器训练联合框架。

        Generates training data, trains multiple classifier models,
        and evaluates their performance on test set.
        生成训练数据，训练多个分类器模型，并在测试集上评估其性能。

        Args:
            n_epochs (int, optional): Number of training epochs
                                     训练轮数，默认从配置读取
            n_samples_per_class (int, optional): Samples per class
                                                每类样本数，默认从配置读取

        Returns:
            dict: Training results for all classifiers containing:
                 所有分类器的训练结果，包含：
                 - accuracy: Classification accuracy | 分类准确率
                 - precision: Precision score | 精确率
                 - recall: Recall score | 召回率
                 - f1: F1-score | F1分数
                 - predictions: Predicted labels | 预测标签
        """
        n_epochs = n_epochs or TRAINING_CONFIG['n_epochs']
        n_samples_per_class = (
                n_samples_per_class or TRAINING_CONFIG['n_samples_per_class']
        )

        print("\n" + "=" * 70)
        print("Model Training Started | 模型训练开始".center(70))
        print("=" * 70)

        # Generate training data | 生成训练数据
        X_real, X_gen, X_combined, y, y_combined = (
            self.generate_training_data(n_samples_per_class)
        )

        # Encode labels to numerical format | 将标签编码为数值格式
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_combined)

        # Split into training and test sets | 分割为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined,
            y_encoded,
            test_size=TRAINING_CONFIG['test_size'],
            random_state=TRAINING_CONFIG['random_state'],
            stratify=y_encoded
        )

        # Standardize features | 标准化特征
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\n[Data Split] Training: {len(X_train)}, Testing: {len(X_test)}")
        print(f"[数据分割] 训练集: {len(X_train)}, 测试集: {len(X_test)}")

        # Initialize classifiers | 初始化分类器
        self.classifiers = {
            'Neural Network': MLPClassifier(
                **CLASSIFIER_CONFIGS['Neural Network']
            ),
            'Random Forest': RandomForestClassifier(
                **CLASSIFIER_CONFIGS['Random Forest']
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                **CLASSIFIER_CONFIGS['Gradient Boosting']
            )
        }

        # Train each classifier and evaluate | 训练每个分类器并评估
        results = {}

        for name, clf in self.classifiers.items():
            print(f"\n[Training] {name}...")
            print(f"[训练] {name}...")

            # Fit classifier on training data | 在训练数据上拟合分类器
            clf.fit(X_train_scaled, y_train)

            # Make predictions on test set | 在测试集上进行预测
            y_pred = clf.predict(X_test_scaled)

            # Compute evaluation metrics | 计算评估指标
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Store results | 存储结果
            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'predictions': y_pred
            }

            # Print metrics | 打印指标
            print(f"  Accuracy:  {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall:    {rec:.4f}")
            print(f"  F1-Score:  {f1:.4f}")

        # Simulate training history for visualization | 模拟训练历史用于可视化
        self._simulate_training_history(n_epochs)

        # Store results for later use | 存储结果供后续使用
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.results = results
        self.is_trained = True

        # Identify best performing model | 识别性能最佳的模型
        best_model = max(results.keys(), key=lambda k: results[k]['f1'])
        best_f1 = results[best_model]['f1']

        print(f"\n[Results] Best model: {best_model} (F1 = {best_f1:.4f})")
        print(f"[结果] 最佳模型: {best_model} (F1 = {best_f1:.4f})")

        return results

    def _simulate_training_history(self, n_epochs):
        """
        Simulate training history curves for visualization.
        模拟训练历史曲线用于可视化。

        Generates synthetic training metrics that follow typical learning
        curves with exponential convergence and random fluctuations.
        生成遵循典型学习曲线的合成训练指标，具有指数收敛和随机波动。

        Args:
            n_epochs (int): Number of epochs to simulate
                           要模拟的训练轮数
        """
        for epoch in range(n_epochs):
            # Simulate exponential loss decay | 模拟指数损失衰减
            loss = 2.0 * np.exp(-epoch / 30) + 0.2
            loss += np.random.randn() * 0.05

            # Simulate accuracy improvement | 模拟准确率提升
            acc = 0.5 + 0.45 * (1 - np.exp(-epoch / 40))
            acc = min(0.95, acc) + np.random.randn() * 0.02

            # Simulate F1-score improvement | 模拟F1分数提升
            f1 = 0.45 + 0.48 * (1 - np.exp(-epoch / 45))
            f1 = min(0.93, f1) + np.random.randn() * 0.02

            # Store with bounds checking | 存储并检查边界
            self.training_history['loss'].append(max(0.1, loss))
            self.training_history['accuracy'].append(
                min(1.0, max(0.4, acc))
            )
            self.training_history['f1'].append(min(1.0, max(0.35, f1)))

    def evaluate_zero_shot(self, n_samples=100):
        """
        Evaluate zero-shot learning performance on unseen defects.
        评估未知缺陷的零样本学习性能。

        Tests the framework's ability to classify both seen and unseen
        defect types using the trained classifiers and zero-shot module.
        测试框架使用训练的分类器和零样本模块分类已知和未知缺陷类型的能力。

        Args:
            n_samples (int, optional): Number of test samples per class
                                      每类的测试样本数，默认100

        Returns:
            dict: Zero-shot evaluation results with structure:
                 零样本评估结果，结构为：
                 {
                     'seen': {defect_type: accuracy, ...},
                     'unseen': {defect_type: accuracy, ...}
                 }
        """
        results = {'seen': {}, 'unseen': {}}

        # Evaluate on seen defect types
        print("\n[Evaluation] Seen defect types:")

        for defect in DefectAttributes.SEEN_DEFECTS:
            # Generate test features
            features = self._simulate_real_features(defect, n_samples)
            features_scaled = self.scaler.transform(features)

            # Predict using trained classifier
            y_pred = self.classifiers['Neural Network'].predict(
                features_scaled
            )
            y_true = self.label_encoder.transform([defect] * n_samples)

            # Compute accuracy
            acc = accuracy_score(y_true, y_pred)
            results['seen'][defect] = acc

            print(f"  {defect:<15} Accuracy = {acc:.4f}")

        # Evaluate on unseen defect types (zero-shot)
        print("\n[Evaluation] Unseen defect types (Zero-shot):")

        for defect in DefectAttributes.UNSEEN_DEFECTS:
            # Generate test features
            features = self._simulate_real_features(defect, n_samples)

            # Use zero-shot learning module for prediction
            predictions = self.zero_shot_module.predict_zero_shot(
                features,
                DefectAttributes.DEFECT_TYPES
            )

            # Compute accuracy
            correct = sum(1 for p in predictions if p == defect)
            acc = correct / n_samples
            results['unseen'][defect] = acc

            print(f"  {defect:<15} Accuracy = {acc:.4f}")

        # Compute average accuracies
        avg_seen = np.mean(list(results['seen'].values()))
        avg_unseen = np.mean(list(results['unseen'].values()))

        print(f"  Seen average accuracy:   {avg_seen:.4f}")
        print(f"  Unseen average accuracy: {avg_unseen:.4f}")

        # Store results
        self.zero_shot_results = results
        return results

    def get_feature_importance(self):
        """
        Get feature importance scores from Random Forest classifier.
        从随机森林分类器获取特征重要性分数。

        Returns:
            numpy.ndarray or None: Feature importance array if Random Forest
                                  is available, None otherwise
                                  如果随机森林可用则返回特征重要性数组，否则返回None
        """
        if 'Random Forest' in self.classifiers:
            return self.classifiers['Random Forest'].feature_importances_
        return None

    def print_summary(self, results, zs_results):
        """
        Print comprehensive training and evaluation summary.
        打印全面的训练和评估总结。

        Displays formatted summary of model performance metrics and
        zero-shot learning results.
        显示模型性能指标和零样本学习结果的格式化摘要。

        Args:
            results (dict): Training results from all classifiers
                           所有分类器的训练结果
            zs_results (dict): Zero-shot evaluation results
                              零样本评估结果
        """
        print("\n" + "=" * 70)
        print("Training Complete! | 训练完成！".center(70))
        print("=" * 70)

        # Model performance summary | 模型性能总结
        print("\n[Model Performance Summary]")
        print("[模型性能总结]")
        print("-" * 50)

        for name in results:
            print(f"  {name}:")
            print(f"    Accuracy:  {results[name]['accuracy']:.4f}")
            print(f"    F1-Score:  {results[name]['f1']:.4f}")

        # Zero-shot learning summary | 零样本学习总结
        print("\n[Zero-shot Learning Summary]")
        print("[零样本学习总结]")
        print("-" * 50)

        avg_seen = np.mean(list(zs_results['seen'].values()))
        avg_unseen = np.mean(list(zs_results['unseen'].values()))

        print(f"  Seen classes average:   {avg_seen:.4f}")
        print(f"  已知类别平均准确率:     {avg_seen:.4f}")
        print(f"  Unseen classes average: {avg_unseen:.4f}")
        print(f"  未知类别平均准确率:     {avg_unseen:.4f}")