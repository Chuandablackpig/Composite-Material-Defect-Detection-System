# -*- coding: utf-8 -*-
"""
Main Entry Point | 主程序入口

Composite Material Defect Detection and Classification System
复合材料缺陷检测与分类系统

This is the main entry point for the defect detection system that combines
diffusion models with zero-shot learning for composite material inspection.
这是缺陷检测系统的主入口程序，结合扩散模型和零样本学习用于复合材料检测。

The system performs the following pipeline:
系统执行以下流程：
1. Framework initialization | 框架初始化
2. Model training with data augmentation | 使用数据增强训练模型
3. Zero-shot evaluation on unseen defects | 对未知缺陷进行零样本评估
4. Result visualization | 结果可视化
5. Performance summary | 性能总结

Usage:
    python main.py
"""

from framework import DiffusionZeroShotFramework
from visualizer import DefectVisualizer
from utils import print_banner, print_header


def main():
    """
    Main execution function for the defect detection system.
    缺陷检测系统的主执行函数。

    Orchestrates the complete workflow including framework initialization,
    model training, zero-shot evaluation, visualization, and summary reporting.
    协调完整的工作流程，包括框架初始化、模型训练、零样本评估、可视化和总结报告。

    The pipeline consists of five main stages:
    流程包含五个主要阶段：

    1. Framework Initialization | 框架初始化
       - Creates DiffusionZeroShotFramework instance
       - 创建DiffusionZeroShotFramework实例

    2. Model Training | 模型训练
       - Generates training data using diffusion model
       - Trains multiple classifiers
       - 使用扩散模型生成训练数据
       - 训练多个分类器

    3. Zero-shot Evaluation | 零样本评估
       - Tests on seen defect types
       - Evaluates on unseen defect types
       - 测试已知缺陷类型
       - 评估未知缺陷类型

    4. Visualization | 可视化
       - Generates training curves
       - Creates comparison plots
       - Produces t-SNE visualizations
       - 生成训练曲线
       - 创建对比图表
       - 生成t-SNE可视化

    5. Summary Report | 总结报告
       - Prints comprehensive performance metrics
       - 打印全面的性能指标

    Returns:
        tuple: (framework, results)
            - framework (DiffusionZeroShotFramework): Trained framework instance
                                                      训练好的框架实例
            - results (dict): Training results from all classifiers
                             所有分类器的训练结果

    Example:
        >>> framework, results = main()
        >>> print(f"Best F1-Score: {max(r['f1'] for r in results.values()):.4f}")
    """

    # Display system banner | 显示系统标题
    print_banner()

    # =========================================================================
    # Step 1: Framework Initialization | 步骤1：框架初始化
    # =========================================================================
    print_header("Step 1: Framework Initialization")
    print_header("步骤1：框架初始化")

    framework = DiffusionZeroShotFramework()

    # =========================================================================
    # Step 2: Model Training | 步骤2：模型训练
    # =========================================================================
    print_header("Step 2: Model Training")
    print_header("步骤2：模型训练")

    results = framework.train()

    # =========================================================================
    # Step 3: Zero-shot Learning Evaluation | 步骤3：零样本学习评估
    # =========================================================================
    print_header("Step 3: Zero-shot Learning Evaluation")
    print_header("步骤3：零样本学习评估")

    zs_results = framework.evaluate_zero_shot()

    # =========================================================================
    # Step 4: Result Visualization | 步骤4：结果可视化
    # =========================================================================
    print_header("Step 4: Result Visualization")
    print_header("步骤4：结果可视化")

    visualizer = DefectVisualizer()
    visualizer.generate_all_plots(framework, results, zs_results)

    # =========================================================================
    # Step 5: Performance Summary | 步骤5：性能总结
    # =========================================================================
    print_header("Step 5: Performance Summary")
    print_header("步骤5：性能总结")

    framework.print_summary(results, zs_results)

    # Print completion message | 打印完成信息
    print("\n" + "=" * 70)
    print("System Execution Completed Successfully!".center(70))
    print("系统执行成功完成！".center(70))
    print("=" * 70 + "\n")

    return framework, results


if __name__ == "__main__":
    """
    Entry point when script is executed directly.
    脚本直接执行时的入口点。

    Executes the main function and stores the returned framework and results
    for potential interactive analysis.
    执行主函数并存储返回的框架和结果以供潜在的交互式分析。
    """
    framework, results = main()

    # Optional: Print quick summary | 可选：打印快速摘要
    print("\n[Quick Access] Framework and results are available for further analysis.")
    print("[快速访问] 框架和结果可用于进一步分析。")
    print(f"  - framework: Trained system instance | 训练好的系统实例")
    print(f"  - results: Model performance metrics | 模型性能指标")