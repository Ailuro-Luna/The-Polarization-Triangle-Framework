import os
import numpy as np
import matplotlib.pyplot as plt

def compare_morality_activation(morality_rates, base_dir):
    """
    绘制不同道德化率下自我激活和社会影响的比较图
    
    参数:
    morality_rates -- 道德化率列表
    base_dir -- 基础目录
    """
    # 创建保存比较图的目录
    comparison_dir = os.path.join(base_dir, "comparison")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # 分别存储每个道德化率下的数据
    all_data = {}
    
    for mor_rate in morality_rates:
        folder_path = os.path.join(base_dir, f"morality_rate_{mor_rate:.1f}")
        csv_path = os.path.join(folder_path, "activation_data.csv")
        
        # 读取CSV数据
        data = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding=None)
        all_data[mor_rate] = data
    
    # 创建比较图
    plt.figure(figsize=(16, 12))
    
    # 1. 不同道德化率下的自我激活分布
    plt.subplot(2, 2, 1)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.hist(data['self_activation'], bins=20, alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Self Activation')
    plt.ylabel('Count')
    plt.title('Self Activation Distribution Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 不同道德化率下的社会影响分布
    plt.subplot(2, 2, 2)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.hist(data['social_influence'], bins=20, alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Social Influence')
    plt.ylabel('Count')
    plt.title('Social Influence Distribution Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 不同道德化率下的总激活分布
    plt.subplot(2, 2, 3)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.hist(data['total_activation'], bins=20, alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Total Activation')
    plt.ylabel('Count')
    plt.title('Total Activation Distribution Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 不同道德化率下的意见分布
    plt.subplot(2, 2, 4)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.hist(data['opinion'], bins=20, alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Opinion')
    plt.ylabel('Count')
    plt.title('Opinion Distribution Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Activation Components Across Morality Rates')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存比较图
    comparison_path = os.path.join(comparison_dir, "activation_distribution_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    
    # 创建散点图比较
    plt.figure(figsize=(16, 12))
    
    # 1. 自我激活 vs 意见
    plt.subplot(2, 2, 1)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.scatter(data['opinion'], data['self_activation'], alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Opinion')
    plt.ylabel('Self Activation')
    plt.title('Self Activation vs Opinion Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 社会影响 vs 意见
    plt.subplot(2, 2, 2)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.scatter(data['opinion'], data['social_influence'], alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Opinion')
    plt.ylabel('Social Influence')
    plt.title('Social Influence vs Opinion Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 总激活 vs 意见
    plt.subplot(2, 2, 3)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.scatter(data['opinion'], data['total_activation'], alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Opinion')
    plt.ylabel('Total Activation')
    plt.title('Total Activation vs Opinion Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 自我激活 vs 社会影响
    plt.subplot(2, 2, 4)
    for mor_rate in morality_rates:
        data = all_data[mor_rate]
        plt.scatter(data['self_activation'], data['social_influence'], alpha=0.5, label=f"Rate={mor_rate:.1f}")
    
    plt.xlabel('Self Activation')
    plt.ylabel('Social Influence')
    plt.title('Self Activation vs Social Influence Across Morality Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Activation Relationships Across Morality Rates')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存散点图比较
    scatter_path = os.path.join(comparison_dir, "activation_scatter_comparison.png")
    plt.savefig(scatter_path)
    plt.close()