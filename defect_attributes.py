# -*- coding: utf-8 -*-
"""
Defect Attributes Module | 缺陷属性模块

This module defines attributes and characteristics for different defect types
in composite materials with their corresponding weights.
本模块定义复合材料中不同缺陷类型的属性特征及其对应权重。

Each defect type is characterized by a set of weighted attributes that
describe its visual and structural properties.
每种缺陷类型由一组加权属性表征，描述其视觉和结构特性。
"""

import numpy as np
from config import DEFECT_TYPES, SEEN_DEFECTS, UNSEEN_DEFECTS


class DefectAttributes:
    """
    Defect Attributes Manager | 缺陷属性管理器

    Manages defect type definitions and their weighted attribute vectors
    for feature extraction and classification tasks.
    管理缺陷类型定义及其加权属性向量，用于特征提取和分类任务。
    """

    # Defect type categories | 缺陷类型分类
    DEFECT_TYPES = DEFECT_TYPES
    SEEN_DEFECTS = SEEN_DEFECTS
    UNSEEN_DEFECTS = UNSEEN_DEFECTS

    # Attribute weights for each defect type | 各缺陷类型的属性权重
    ATTRIBUTE_WEIGHTS = {
        'Crack': {
            'linear_shape': 0.18,
            'aligned_orientation': 0.13,
            'dark_intensity': 0.12,
            'thin_width': 0.11,
            'high_aspect_ratio': 0.12,
            'edge_sharpness': 0.10,
            'surface_penetration': 0.12,
            'irregular_path': 0.12
        },
        'Scratch': {
            'linear_shape': 0.16,
            'surface_level': 0.14,
            'light_intensity': 0.13,
            'thin_width': 0.12,
            'continuous_path': 0.11,
            'uniform_depth': 0.12,
            'directional_pattern': 0.11,
            'shallow_penetration': 0.11
        },
        'Inclusion': {
            'circular_shape': 0.17,
            'embedded_depth': 0.15,
            'contrast_intensity': 0.13,
            'isolated_occurrence': 0.12,
            'defined_boundary': 0.11,
            'foreign_material': 0.11,
            'variable_size': 0.11,
            'random_distribution': 0.10
        },
        'Porosity': {
            'circular_shape': 0.15,
            'dark_intensity': 0.14,
            'clustered_pattern': 0.13,
            'small_size': 0.12,
            'subsurface_location': 0.12,
            'void_structure': 0.12,
            'random_distribution': 0.11,
            'variable_density': 0.11
        },
        'Delamination': {
            'planar_shape': 0.16,
            'layered_structure': 0.15,
            'interface_location': 0.14,
            'large_area': 0.13,
            'irregular_boundary': 0.11,
            'subsurface_depth': 0.11,
            'stress_induced': 0.10,
            'propagating_nature': 0.10
        },
        'Fiber_Break': {
            'linear_discontinuity': 0.17,
            'fiber_alignment': 0.15,
            'localized_damage': 0.14,
            'sharp_edges': 0.12,
            'stress_concentration': 0.12,
            'structural_weakness': 0.11,
            'visible_fracture': 0.10,
            'load_direction': 0.09
        }
    }

    @classmethod
    def get_attribute_vector(cls, defect_type):
        """
        Get numerical attribute vector for a specific defect type.
        获取指定缺陷类型的数值属性向量。

        Converts the weighted attributes dictionary into a fixed-length
        numerical vector for machine learning tasks.
        将加权属性字典转换为固定长度的数值向量，用于机器学习任务。

        Args:
            defect_type (str): Name of the defect type
                              缺陷类型名称

        Returns:
            numpy.ndarray: Attribute weight vector of shape (n_attributes,)
                          形状为(n_attributes,)的属性权重向量

        Raises:
            ValueError: If defect_type is not recognized
                       如果缺陷类型不被识别
        """

        if not cls.validate_defect_type(defect_type):
            raise ValueError(
                f"Unknown defect type: {defect_type}. "
                f"Valid types: {cls.DEFECT_TYPES}"
            )

        attrs = cls.ATTRIBUTE_WEIGHTS.get(defect_type, {})
        all_attrs = cls.get_all_attributes()

        # Build fixed-length vector | 构建固定长度向量
        vector = [attrs.get(attr, 0) for attr in all_attrs]

        return np.array(vector)

    @classmethod
    def get_all_attributes(cls):
        """
        Get all unique attribute names across defect types.
        获取所有缺陷类型的唯一属性名称。

        Returns:
            list: Sorted list of unique attribute names
                 排序后的唯一属性名称列表
        """

        all_attrs = set()
        for weights_dict in cls.ATTRIBUTE_WEIGHTS.values():
            all_attrs.update(weights_dict.keys())

        return sorted(list(all_attrs))

    @classmethod
    def get_defect_info(cls, defect_type):
        """
        Get comprehensive information for a specific defect type.
        获取指定缺陷类型的全面信息。

        Args:
            defect_type (str): Name of the defect type
                              缺陷类型名称

        Returns:
            dict: Dictionary containing defect information with keys:
                 包含缺陷信息的字典，包含以下键：
                 - name: Defect type name | 缺陷类型名称
                 - attributes: Attribute weights dict | 属性权重字典
                 - is_seen: Whether in training set | 是否在训练集中
                 - attribute_vector: Numerical vector | 数值向量
        """

        return {
            'name': defect_type,
            'attributes': cls.ATTRIBUTE_WEIGHTS.get(defect_type, {}),
            'is_seen': defect_type in cls.SEEN_DEFECTS,
            'attribute_vector': cls.get_attribute_vector(defect_type)
        }

    @classmethod
    def validate_defect_type(cls, defect_type):
        """
        Validate if a defect type exists in the system.
        验证缺陷类型是否存在于系统中。

        Args:
            defect_type (str): Name of the defect type to validate
                              要验证的缺陷类型名称

        Returns:
            bool: True if defect type is valid, False otherwise
                 如果缺陷类型有效返回True，否则返回False
        """
        return defect_type in cls.DEFECT_TYPES

    @classmethod
    def get_attribute_count(cls, defect_type):
        """
        Get the number of attributes for a defect type.
        获取缺陷类型的属性数量。

        Args:
            defect_type (str): Name of the defect type
                              缺陷类型名称

        Returns:
            int: Number of attributes
                属性数量
        """
        return len(cls.ATTRIBUTE_WEIGHTS.get(defect_type, {}))

    @classmethod
    def get_weight_sum(cls, defect_type):
        """
        Calculate the sum of attribute weights for a defect type.
        计算缺陷类型的属性权重总和。

        Args:
            defect_type (str): Name of the defect type
                              缺陷类型名称

        Returns:
            float: Sum of attribute weights (should be ~1.0)
                  属性权重总和（应约为1.0）
        """
        attrs = cls.ATTRIBUTE_WEIGHTS.get(defect_type, {})
        return sum(attrs.values())

    @classmethod
    def is_seen_defect(cls, defect_type):
        """
        Check if defect type is in the training set.
        检查缺陷类型是否在训练集中。

        Args:
            defect_type (str): Name of the defect type
                              缺陷类型名称

        Returns:
            bool: True if defect is in seen set
                 如果缺陷在已知集合中返回True
        """
        return defect_type in cls.SEEN_DEFECTS

    @classmethod
    def get_defect_summary(cls):
        """
        Get summary statistics for all defect types.
        获取所有缺陷类型的摘要统计信息。

        Returns:
            dict: Summary information including counts and lists
                 包含计数和列表的摘要信息
        """
        return {
            'total_defects': len(cls.DEFECT_TYPES),
            'seen_defects': len(cls.SEEN_DEFECTS),
            'unseen_defects': len(cls.UNSEEN_DEFECTS),
            'total_attributes': len(cls.get_all_attributes()),
            'defect_list': cls.DEFECT_TYPES,
            'seen_list': cls.SEEN_DEFECTS,
            'unseen_list': cls.UNSEEN_DEFECTS
        }

    @classmethod
    def compare_defects(cls, defect_type1, defect_type2):
        """
        Compare attribute vectors between two defect types.
        比较两种缺陷类型之间的属性向量。

        Args:
            defect_type1 (str): First defect type name
                               第一个缺陷类型名称
            defect_type2 (str): Second defect type name
                               第二个缺陷类型名称

        Returns:
            dict: Comparison results including cosine similarity
                 比较结果，包括余弦相似度
        """
        vec1 = cls.get_attribute_vector(defect_type1)
        vec2 = cls.get_attribute_vector(defect_type2)

        # Compute cosine similarity | 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        similarity = dot_product / (norm_product + 1e-8)

        # Compute Euclidean distance | 计算欧氏距离
        distance = np.linalg.norm(vec1 - vec2)

        return {
            'defect_1': defect_type1,
            'defect_2': defect_type2,
            'cosine_similarity': float(similarity),
            'euclidean_distance': float(distance)
        }