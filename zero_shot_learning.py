# -*- coding: utf-8 -*-
"""
Zero-shot Learning Module | 零样本学习模块

This module implements zero-shot learning for defect classification using
visual-semantic joint embedding space.
本模块实现基于视觉-语义联合嵌入空间的零样本缺陷分类。

The module creates a joint embedding space where visual features and
semantic attributes are projected, enabling classification of unseen defect types.
该模块创建联合嵌入空间，将视觉特征和语义属性投影其中，实现未知缺陷类型的分类。
"""

import numpy as np
from defect_attributes import DefectAttributes
from config import ZERO_SHOT_CONFIG


class ZeroShotLearningModule:
    """
    Zero-shot Learning Module for Defect Classification
    零样本学习缺陷分类模块

    Implements visual-semantic joint embedding for classifying unseen
    defect types based on their attribute descriptions.
    实现视觉-语义联合嵌入，基于属性描述对未知缺陷类型进行分类。

    The module uses multi-head self-attention for visual encoding and
    attribute-weighted semantic embedding for semantic representation.
    该模块使用多头自注意力进行视觉编码，使用属性加权语义嵌入进行语义表示。

    Attributes:
        visual_dim (int): Dimension of visual features
                         视觉特征维度
        semantic_dim (int): Dimension of semantic features
                           语义特征维度
        embedding_dim (int): Dimension of joint embedding space
                            联合嵌入空间维度
        visual_projection (numpy.ndarray): Visual feature projection matrix
                                          视觉特征投影矩阵
        semantic_projection (numpy.ndarray): Semantic feature projection matrix
                                            语义特征投影矩阵
        class_semantics (dict): Semantic embeddings for each defect class
                               每个缺陷类别的语义嵌入
    """

    def __init__(self, visual_dim=None, semantic_dim=None, embedding_dim=None):
        """
        Initialize the zero-shot learning module.
        初始化零样本学习模块。

        Sets up projection matrices for visual and semantic features,
        and builds semantic embeddings for all defect classes.
        设置视觉和语义特征的投影矩阵，并构建所有缺陷类别的语义嵌入。

        Args:
            visual_dim (int, optional): Visual feature dimension
                                       视觉特征维度，默认从配置读取
            semantic_dim (int, optional): Semantic feature dimension
                                         语义特征维度，默认从配置读取
            embedding_dim (int, optional): Joint embedding space dimension
                                          联合嵌入空间维度，默认从配置读取
        """
        # Load dimensions from configuration | 从配置加载维度参数
        self.visual_dim = visual_dim or ZERO_SHOT_CONFIG['visual_dim']
        self.semantic_dim = semantic_dim or ZERO_SHOT_CONFIG['semantic_dim']
        self.embedding_dim = embedding_dim or ZERO_SHOT_CONFIG['embedding_dim']

        # Initialize visual feature projection matrix | 初始化视觉特征投影矩阵
        # Simulates ViT (Vision Transformer) output projection
        # 模拟ViT（视觉Transformer）输出投影
        self.visual_projection = np.random.randn(
            self.visual_dim, self.embedding_dim
        ) * 0.1

        # Initialize semantic feature projection matrix | 初始化语义特征投影矩阵
        # Simulates BERT-like semantic encoder projection
        # 模拟类BERT语义编码器投影
        self.semantic_projection = np.random.randn(
            self.semantic_dim, self.embedding_dim
        ) * 0.1

        # Build semantic embeddings for all defect classes | 构建所有缺陷类别的语义嵌入
        self.attribute_embeddings = None
        self.class_semantics = {}
        self._build_semantic_embeddings()

        print(f"[Zero-shot Learning] Initialized: "
              f"Visual {self.visual_dim}D -> Embedding {self.embedding_dim}D")
        print(f"[零样本学习] 初始化完成: "
              f"视觉 {self.visual_dim}D -> 嵌入 {self.embedding_dim}D")

    def _build_semantic_embeddings(self):
        """
        Build semantic embeddings for all defect classes.
        构建所有缺陷类别的语义嵌入。

        Creates semantic vectors for each defect type by combining
        attribute embeddings weighted by their importance scores.
        通过组合按重要性分数加权的属性嵌入，为每种缺陷类型创建语义向量。

        The semantic vector for class c is computed as:
        类别c的语义向量计算为：
        s_c = Σ w_i * e_i
        where w_i are attribute weights and e_i are attribute embeddings.
        其中w_i是属性权重，e_i是属性嵌入。
        """
        # Get all unique attributes | 获取所有唯一属性
        attributes = DefectAttributes.get_all_attributes()
        n_attrs = len(attributes)

        # Initialize attribute embedding matrix | 初始化属性嵌入矩阵
        self.attribute_embeddings = np.random.randn(
            n_attrs, self.semantic_dim
        ) * 0.5

        # Compute semantic vector for each defect type | 计算每种缺陷类型的语义向量
        self.class_semantics = {}
        for defect_type in DefectAttributes.DEFECT_TYPES:
            # Get attribute weights for this defect type | 获取此缺陷类型的属性权重
            weights = DefectAttributes.get_attribute_vector(defect_type)

            # Compute weighted sum of attribute embeddings | 计算属性嵌入的加权和
            semantic_vec = weights @ self.attribute_embeddings

            # Store semantic vector | 存储语义向量
            self.class_semantics[defect_type] = semantic_vec

    def encode_visual(self, features):
        """
        Encode visual features into joint embedding space.
        将视觉特征编码到联合嵌入空间。

        Applies multi-head self-attention mechanism followed by projection
        to transform raw visual features into the joint embedding space.
        应用多头自注意力机制，然后投影，将原始视觉特征转换到联合嵌入空间。

        Args:
            features (numpy.ndarray): Input visual features of shape (n_samples, visual_dim)
                                     形状为(n_samples, visual_dim)的输入视觉特征

        Returns:
            numpy.ndarray: Visual embeddings of shape (n_samples, embedding_dim)
                          形状为(n_samples, embedding_dim)的视觉嵌入
        """
        # Apply multi-head self-attention | 应用多头自注意力
        attention_output = self._self_attention(features)

        # Project to joint embedding space | 投影到联合嵌入空间
        embedded = attention_output @ self.visual_projection

        return embedded

    def _self_attention(self, x, n_heads=None):
        """
        Compute multi-head self-attention for visual features.
        计算视觉特征的多头自注意力。

        Implements scaled dot-product attention mechanism:
        实现缩放点积注意力机制：
        Attention(Q, K, V) = softmax(QK^T / √d_k) V

        Args:
            x (numpy.ndarray): Input features of shape (n_samples, feature_dim)
                              形状为(n_samples, feature_dim)的输入特征
            n_heads (int, optional): Number of attention heads
                                    注意力头数，默认从配置读取

        Returns:
            numpy.ndarray: Attention output of same shape as input
                          与输入形状相同的注意力输出
        """
        n_heads = n_heads or ZERO_SHOT_CONFIG['n_heads']
        d_k = x.shape[1] // n_heads

        # Simplified attention computation | 简化的注意力计算
        # In practice, Q, K, V would be separate projections of x
        # 实际应用中，Q、K、V应该是x的独立投影
        Q = x
        K = x
        V = x

        # Compute attention scores | 计算注意力分数
        scores = Q @ K.T / np.sqrt(d_k)

        # Apply softmax to get attention weights | 应用softmax获得注意力权重
        attention_weights = self._softmax(scores)

        # Apply attention weights to values | 将注意力权重应用于值
        output = attention_weights @ V

        return output

    def _softmax(self, x):
        """
        Compute softmax function with numerical stability.
        计算具有数值稳定性的softmax函数。

        Uses the log-sum-exp trick to prevent numerical overflow.
        使用log-sum-exp技巧防止数值溢出。

        Args:
            x (numpy.ndarray): Input array
                              输入数组

        Returns:
            numpy.ndarray: Softmax probabilities of same shape as input
                          与输入形状相同的softmax概率
        """
        # Subtract max for numerical stability | 减去最大值以保持数值稳定性
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

        # Normalize | 归一化
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def encode_semantic(self, defect_type):
        """
        Encode semantic attributes into joint embedding space.
        将语义属性编码到联合嵌入空间。

        Projects the semantic attribute vector of a defect type into
        the joint embedding space for similarity comparison.
        将缺陷类型的语义属性向量投影到联合嵌入空间以进行相似度比较。

        Args:
            defect_type (str): Name of the defect type
                              缺陷类型名称

        Returns:
            numpy.ndarray: Semantic embedding of shape (embedding_dim,)
                          形状为(embedding_dim,)的语义嵌入
        """
        # Get pre-computed semantic vector | 获取预计算的语义向量
        semantic_vec = self.class_semantics.get(defect_type)

        if semantic_vec is None:
            # Return zero vector for unknown classes | 对未知类别返回零向量
            return np.zeros(self.embedding_dim)

        # Project to joint embedding space | 投影到联合嵌入空间
        embedded = semantic_vec @ self.semantic_projection

        return embedded

    def compute_similarity(self, visual_embedding, class_name):
        """
        Compute cosine similarity between visual and semantic embeddings.
        计算视觉嵌入和语义嵌入之间的余弦相似度。

        Measures the similarity in the joint embedding space to determine
        how well a visual feature matches a semantic class description.
        在联合嵌入空间中测量相似度，以确定视觉特征与语义类别描述的匹配程度。

        Args:
            visual_embedding (numpy.ndarray): Visual feature embedding
                                             视觉特征嵌入
            class_name (str): Name of the class to compare
                             要比较的类别名称

        Returns:
            float: Cosine similarity score in range [-1, 1]
                  范围在[-1, 1]的余弦相似度分数
        """
        # Get semantic embedding for the class | 获取类别的语义嵌入
        semantic_embedding = self.encode_semantic(class_name)

        # Compute dot product | 计算点积
        similarity = np.dot(visual_embedding, semantic_embedding)

        # Normalize by vector magnitudes (cosine similarity) | 按向量幅度归一化（余弦相似度）
        norm_visual = np.linalg.norm(visual_embedding)
        norm_semantic = np.linalg.norm(semantic_embedding)
        similarity /= (norm_visual * norm_semantic + 1e-8)

        return similarity

    def predict_zero_shot(self, visual_features, candidate_classes):
        """
        Perform zero-shot classification on visual features.
        对视觉特征执行零样本分类。

        Classifies visual features by finding the most similar semantic
        class in the joint embedding space, enabling classification of
        unseen defect types.
        通过在联合嵌入空间中找到最相似的语义类别来对视觉特征进行分类，
        实现对未知缺陷类型的分类。

        Args:
            visual_features (numpy.ndarray): Visual features to classify
                                            要分类的视觉特征，形状(n_samples, visual_dim)
            candidate_classes (list): List of candidate class names
                                     候选类别名称列表
        """
        # Encode visual features | 编码视觉特征
        visual_embeddings = self.encode_visual(visual_features)

        predictions = []

        # Classify each sample | 对每个样本进行分类
        for feat in visual_embeddings:
            similarities = {}

            # Compute similarity to all candidate classes | 计算与所有候选类别的相似度
            for cls in candidate_classes:
                sim = self.compute_similarity(feat, cls)
                similarities[cls] = sim

            # Predict class with the highest similarity | 预测相似度最高的类别
            pred = max(similarities, key=similarities.get)
            predictions.append(pred)

        return predictions

    def get_class_prototypes(self):
        """
        Get semantic prototypes for all defect classes.
        获取所有缺陷类别的语义原型。

        Returns:
            dict: Dictionary mapping class names to semantic embeddings
                 将类别名称映射到语义嵌入的字典
        """
        prototypes = {}
        for defect_type in DefectAttributes.DEFECT_TYPES:
            prototypes[defect_type] = self.encode_semantic(defect_type)
        return prototypes

    def compute_embedding_quality(self):
        """
        Compute quality metrics for the embedding space.
        计算嵌入空间的质量指标。

        Returns:
            dict: Dictionary containing embedding quality metrics
                 包含嵌入质量指标的字典
        """
        prototypes = self.get_class_prototypes()

        # Compute pairwise distances | 计算成对距离
        class_names = list(prototypes.keys())
        n_classes = len(class_names)
        distances = np.zeros((n_classes, n_classes))

        for i, cls1 in enumerate(class_names):
            for j, cls2 in enumerate(class_names):
                if i != j:
                    emb1 = prototypes[cls1]
                    emb2 = prototypes[cls2]
                    dist = np.linalg.norm(emb1 - emb2)
                    distances[i, j] = dist

        # Compute statistics | 计算统计数据
        mean_distance = np.mean(distances[distances > 0])
        min_distance = np.min(distances[distances > 0])
        max_distance = np.max(distances)

        return {
            'mean_distance': float(mean_distance),
            'min_distance': float(min_distance),
            'max_distance': float(max_distance),
            'n_classes': n_classes
        }