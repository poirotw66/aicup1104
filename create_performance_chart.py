"""
創建效能對比圖表
"""

import matplotlib.pyplot as plt
import numpy as np

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 資料
versions = ['main.py\n(原始)', 'main_fast.py\n(舊版)', 'main_fast.py\n(Ultra Fast)\n✨推薦', 'main_ultra_fast.py\n(Parquet)']
phase2_times = [19.5, 9, 5, 2.5]  # Phase 2 時間（分鐘）
total_times = [21, 10.5, 6.5, 4.5]  # 總時間（分鐘）
speedup = [1.0, 2.0, 3.2, 4.7]  # 相對加速

# 建立圖表
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 圖 1：Phase 2 特徵提取時間
ax1 = axes[0]
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#27ae60']
bars1 = ax1.bar(range(len(versions)), phase2_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Version', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time (minutes)', fontsize=12, fontweight='bold')
ax1.set_title('Phase 2: Feature Extraction Time\n(Main Bottleneck - 93% of total time)', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(versions)))
ax1.set_xticklabels(versions, fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 添加數值標籤
for i, (bar, time) in enumerate(zip(bars1, phase2_times)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.1f} min',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 圖 2：總執行時間
ax2 = axes[1]
bars2 = ax2.bar(range(len(versions)), total_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Version', fontsize=12, fontweight='bold')
ax2.set_ylabel('Time (minutes)', fontsize=12, fontweight='bold')
ax2.set_title('Total Execution Time\n(4.43 million transactions)', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(versions)))
ax2.set_xticklabels(versions, fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# 添加數值標籤
for i, (bar, time) in enumerate(zip(bars2, total_times)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.1f} min',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# 圖 3：相對加速倍數
ax3 = axes[2]
bars3 = ax3.bar(range(len(versions)), speedup, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Version', fontsize=12, fontweight='bold')
ax3.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
ax3.set_title('Relative Speedup\n(vs Original Version)', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(versions)))
ax3.set_xticklabels(versions, fontsize=10)
ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.legend()

# 添加數值標籤
for i, (bar, speed) in enumerate(zip(bars3, speedup)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{speed:.1f}x',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.suptitle('Performance Optimization Results - Alert Account Detection System', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('output/performance_comparison.png', dpi=150, bbox_inches='tight')
print("Performance comparison chart saved to output/performance_comparison.png")

# 建立第二個圖表：Phase 分解
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))

# 各階段時間（以 main.py 為例）
phases = ['Phase 1:\nLoading\nData', 'Phase 2:\nFeature\nEngineering', 
          'Phase 3:\nEDA\nAnalysis', 'Phase 4:\nPattern\nDiscovery', 
          'Phase 5:\nRule\nBuilding', 'Phase 6:\nMaking\nPredictions']
original_times = [1, 19.5, 0.5, 0.9, 0.05, 0.05]
optimized_times = [1, 5, 0.5, 0.9, 0.05, 0.05]

x = np.arange(len(phases))
width = 0.35

bars1 = ax.bar(x - width/2, original_times, width, label='main.py (Original)', 
               color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, optimized_times, width, label='main_fast.py (Ultra Fast)', 
               color='#2ecc71', alpha=0.8, edgecolor='black')

ax.set_xlabel('Phase', fontsize=12, fontweight='bold')
ax.set_ylabel('Time (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Time Breakdown by Phase', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=10)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加數值標籤
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.5:  # 只標記較大的值
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

# 添加總計
total_original = sum(original_times)
total_optimized = sum(optimized_times)
ax.text(0.02, 0.98, f'Total (Original): {total_original:.1f} min', 
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
ax.text(0.02, 0.90, f'Total (Optimized): {total_optimized:.1f} min', 
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
ax.text(0.02, 0.82, f'Speedup: {total_original/total_optimized:.1f}x', 
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))

plt.tight_layout()
plt.savefig('output/phase_breakdown.png', dpi=150, bbox_inches='tight')
print("Phase breakdown chart saved to output/phase_breakdown.png")

plt.close('all')
print("\n✅ All charts created successfully!")

