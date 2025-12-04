# -*- coding: utf-8 -*-
"""
Diffusion Model Module | 扩散模型模块

This module implements a conditional diffusion model for generating
synthetic defect features in composite materials.
本模块实现条件扩散模型，用于生成复合材料的合成缺陷特征。

The model uses cosine noise scheduling and supports conditional generation
based on defect type attributes.
该模型使用余弦噪声调度，支持基于缺陷类型属性的条件生成。
"""

import numpy as np
from defect_attributes import DefectAttributes
from config import DIFFUSION_CONFIG


class DiffusionModelSimulator:
    """
    Conditional Diffusion Model Simulator | 条件扩散模型模拟器

    Implements a diffusion-based generative model for creating realistic
    defect features conditioned on defect type attributes.
    实现基于扩散的生成模型，用于创建基于缺陷类型属性的真实缺陷特征。
    """

    def __init__(self, timesteps=None, beta_start=None, beta_end=None):
        """
        Initialize the diffusion model simulator.
        初始化扩散模型模拟器。

        Sets up noise scheduling parameters and computes cumulative products
        for efficient forward diffusion process.
        设置噪声调度参数并计算累积乘积以实现高效的前向扩散过程。

        Args:
            timesteps (int, optional): Number of diffusion steps
                                      扩散步数，默认从配置读取
            beta_start (float, optional): Starting noise level
                                         起始噪声水平，默认从配置读取
            beta_end (float, optional): Ending noise level
                                       结束噪声水平，默认从配置读取
        """
        # Load default configuration | 加载默认配置
        self.timesteps = timesteps or DIFFUSION_CONFIG['timesteps']
        beta_start = beta_start or DIFFUSION_CONFIG['beta_start']
        beta_end = beta_end or DIFFUSION_CONFIG['beta_end']

        # Compute noise schedule using cosine strategy | 使用余弦策略计算噪声调度
        self.betas = self._cosine_beta_schedule(
            self.timesteps, beta_start, beta_end
        )
        self.alphas = 1 - self.betas
        self.alpha_cumprod = np.cumprod(self.alphas)

        print(f"[Diffusion Model] Initialized: {self.timesteps} steps, "
              f"β ∈ [{beta_start:.4f}, {beta_end:.4f}]")
        print(f"[扩散模型] 初始化完成: {self.timesteps} 步, "
              f"β ∈ [{beta_start:.4f}, {beta_end:.4f}]")

    def _cosine_beta_schedule(self, timesteps, beta_start, beta_end):
        """
        Compute cosine noise schedule for stable training.
        计算余弦噪声调度以实现稳定训练。

        Uses a cosine function to create smooth transitions in noise levels,
        which helps improve training stability and sample quality.
        使用余弦函数创建平滑的噪声水平过渡，有助于提高训练稳定性和样本质量。

        Args:
            timesteps (int): Total number of diffusion steps
                            总扩散步数
            beta_start (float): Minimum beta value
                               最小beta值
            beta_end (float): Maximum beta value
                             最大beta值

        Returns:
            numpy.ndarray: Beta schedule array of shape (timesteps),
                          形状为(timesteps),的beta调度数组
        """
        steps = np.linspace(0, timesteps, timesteps + 1)

        # Cosine schedule for smooth noise transitions | 余弦调度实现平滑噪声过渡
        alpha_bar = np.cos(((steps / timesteps) + 0.008) / 1.008 * np.pi / 2)
        alpha_bar = alpha_bar ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        # Compute betas from alpha_bar | 从alpha_bar计算beta值
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])

        # Clip to valid range | 裁剪到有效范围
        return np.clip(betas, beta_start, beta_end)

    def forward_diffusion(self, x0, t):
        """
        Apply forward diffusion process to add noise.
        应用前向扩散过程添加噪声。

        The forward process gradually adds Gaussian noise to the input data
        according to the noise schedule. Process follows:
        前向过程根据噪声调度逐渐向输入数据添加高斯噪声。过程遵循：
        q(x_t|x_0) = N(x_t; sqrt(α̅_t)x_0, (1-α̅_t)I)

        Args:
            x0 (numpy.ndarray): Clean input data of shape (n_samples, n_features)
                               形状为(n_samples, n_features)的干净输入数据
            t (int): Time step index (0 <= t < timesteps)
                    时间步索引 (0 <= t < timesteps)

        Returns:
            tuple: (x_t, noise)
                - x_t (numpy.ndarray): Noised data at time t
                                      时间步t的噪声数据
                - noise (numpy.ndarray): The noise that was added
                                        添加的噪声

        Raises:
            ValueError: If t is out of valid range
                       如果t超出有效范围
        """
        if t < 0 or t >= self.timesteps:
            raise ValueError(
                f"Time step t={t} out of range [0, {self.timesteps})"
            )

        # Compute noise coefficients | 计算噪声系数
        sqrt_alpha_cumprod = np.sqrt(self.alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod = np.sqrt(1 - self.alpha_cumprod[t])

        # Sample Gaussian noise | 采样高斯噪声
        noise = np.random.randn(*x0.shape)

        # Apply forward diffusion equation | 应用前向扩散方程
        x_t = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

        return x_t, noise

    def generate_defect_features(self, defect_type, n_samples=100,
                                 feature_dim=64):
        """
        Generate synthetic defect features conditioned on defect type.
        基于缺陷类型生成合成缺陷特征。

        This method simulates the reverse diffusion process to generate
        realistic defect features guided by defect type attributes.
        该方法模拟反向扩散过程，生成由缺陷类型属性引导的真实缺陷特征。

        Args:
            defect_type (str): Type of defect to generate
                              要生成的缺陷类型
            n_samples (int, optional): Number of samples to generate
                                      生成的样本数，默认100
            feature_dim (int, optional): Dimension of feature vectors
                                        特征向量维度，默认64

        Returns:
            numpy.ndarray: Generated feature matrix of shape (n_samples, feature_dim)
                          形状为(n_samples, feature_dim)的生成特征矩阵

        Raises:
            ValueError: If defect_type is invalid or feature_dim <= 0
                       如果缺陷类型无效或特征维度<=0

        Example:
            >>> model = DiffusionModelSimulator()
            >>> features = model.generate_defect_features('Crack', n_samples=50)
            >>> print(features.shape)
            (50, 64)
        """
        # Validate inputs | 验证输入
        if not DefectAttributes.validate_defect_type(defect_type):
            raise ValueError(f"Invalid defect type: {defect_type}")
        if feature_dim <= 0:
            raise ValueError(f"Feature dimension must be positive: {feature_dim}")

        # Get defect attributes as conditioning information | 获取缺陷属性作为条件信息
        condition = DefectAttributes.get_attribute_vector(defect_type)
        condition = np.tile(condition, (n_samples, 1))

        # Generate base features from noise | 从噪声生成基础特征
        base_features = np.random.randn(n_samples, feature_dim)

        # Project condition to feature space | 将条件投影到特征空间
        n_attrs = len(DefectAttributes.get_all_attributes())
        cond_projection = np.random.randn(n_attrs, feature_dim)
        cond_features = condition @ cond_projection

        # Combine base and conditional features | 组合基础特征和条件特征
        # Weighted combination for controlled generation | 加权组合实现可控生成
        features = base_features * 0.3 + cond_features * 0.7

        # Add type-specific patterns | 添加类型特定模式
        type_idx = DefectAttributes.DEFECT_TYPES.index(defect_type)
        features[:, :min(8, feature_dim)] += type_idx * 0.5

        return features

    def compute_fid_score(self, real_features, generated_features):
        """
        Compute Fréchet Inception Distance (FID) score.
        计算Fréchet Inception Distance (FID)分数。

        FID measures the similarity between real and generated distributions
        by comparing their mean and covariance in feature space.
        FID通过比较特征空间中的均值和协方差来衡量真实分布和生成分布之间的相似度。

        Formula: FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^0.5)

        Args:
            real_features (numpy.ndarray): Real feature samples
                                          真实特征样本，形状(n_real, n_features)
            generated_features (numpy.ndarray): Generated feature samples
                                               生成的特征样本，形状(n_gen, n_features)

        Returns:
            float: FID score (lower is better, 0 indicates identical distributions)
                  FID分数（越低越好，0表示分布相同）
        """
        # Compute mean vectors | 计算均值向量
        mu_r = np.mean(real_features, axis=0)
        mu_g = np.mean(generated_features, axis=0)

        # Compute covariance matrices | 计算协方差矩阵
        sigma_r = np.cov(real_features.T)
        sigma_g = np.cov(generated_features.T)

        # Mean difference term | 均值差异项
        diff = mu_r - mu_g
        mean_term = np.sum(diff ** 2)

        # Covariance term (simplified) | 协方差项（简化版）
        cov_term = np.trace(
            sigma_r + sigma_g - 2 * np.sqrt(np.abs(sigma_r * sigma_g))
        )

        fid = mean_term + cov_term

        return max(0, fid)

    def get_noise_level(self, t):
        """
        Get the noise level at a specific time step.
        获取特定时间步的噪声水平。

        Args:
            t (int): Time step index
                    时间步索引

        Returns:
            float: Noise level (beta value)
                  噪声水平（beta值）
        """
        if t < 0 or t >= self.timesteps:
            raise ValueError(f"Time step t={t} out of range")
        return float(self.betas[t])

    def get_signal_rate(self, t):
        """
        Get the signal retention rate at a specific time step.
        获取特定时间步的信号保留率。

        Args:
            t (int): Time step index
                    时间步索引

        Returns:
            float: Signal rate (cumulative alpha)
                  信号率（累积alpha值）
        """
        if t < 0 or t >= self.timesteps:
            raise ValueError(f"Time step t={t} out of range")
        return float(self.alpha_cumprod[t])

    def get_schedule_info(self):
        """
        Get complete noise schedule information.
        获取完整的噪声调度信息。

        Returns:
            dict: Dictionary containing schedule parameters
                 包含调度参数的字典
        """
        return {
            'timesteps': self.timesteps,
            'beta_min': float(np.min(self.betas)),
            'beta_max': float(np.max(self.betas)),
            'beta_mean': float(np.mean(self.betas)),
            'alpha_final': float(self.alpha_cumprod[-1])
        }