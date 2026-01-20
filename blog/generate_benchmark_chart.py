"""
Generate benchmark bar chart for the CUDA Vector Addition blog post.
Requires: pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data from proper methodology: CUDA events, pinned memory, averaged reps (Dec 2025)
# GTX 1650 (14 SMs) + 12-thread CPU
n_elements = ['1K', '10K', '100K', '1M', '10M', '100M']
cpu_single = [0.0001, 0.003, 0.033, 1.02, 13.6, 146.0]  # Using 0.0001 for display (actual ~0)
cpu_openmp = [0.002, 0.003, 0.009, 0.394, 12.3, 128.0]
gpu_with_transfer = [0.149, 0.145, 0.139, 1.202, 11.5, 123.0]
gpu_kernel_only = [0.003, 0.003, 0.011, 0.084, 0.82, 8.28]

x = np.arange(len(n_elements))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

# Create bars
bars1 = ax.bar(x - 1.5*width, cpu_single, width, label='CPU (1 thread)', color='#3498db')
bars2 = ax.bar(x - 0.5*width, cpu_openmp, width, label='CPU (12 threads)', color='#85c1e9')
bars3 = ax.bar(x + 0.5*width, gpu_with_transfer, width, label='GPU (w/ transfer)', color='#e74c3c')
bars4 = ax.bar(x + 1.5*width, gpu_kernel_only, width, label='GPU (kernel only)', color='#2ecc71')

# Customize
ax.set_xlabel('Array Size (N elements)', fontsize=12, fontweight='bold')
ax.set_ylabel('Time (ms) - Log Scale', fontsize=12, fontweight='bold')
ax.set_title('Vector Addition Performance: CPU vs GPU\n(GTX 1650, 12-thread CPU)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(n_elements)
ax.legend(loc='upper left')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}' if height >= 0.1 else f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=45)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Add winner indicators
winners = [
    (0, 0, '✓'),   # 1K: CPU single wins
    (2, 1, '✓'),   # 100K: CPU OpenMP wins
    (4, 3, '✓'),   # 10M: GPU kernel wins
]

plt.tight_layout()
plt.savefig('blog/images/benchmark_chart.png', dpi=150, bbox_inches='tight')
plt.savefig('blog/images/benchmark_chart.svg', bbox_inches='tight')
print("Saved: blog/images/benchmark_chart.png and .svg")
plt.show()
