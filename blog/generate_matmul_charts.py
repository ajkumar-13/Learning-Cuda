# generate_matmul_charts.py
# Generate benchmark visualization charts for the matrix multiplication blog
# Run: python generate_matmul_charts.py

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

# Benchmark data from actual runs on GTX 1650 Max-Q
sizes = ['256×256', '512×512', '1024×1024', '2048×2048']
sizes_numeric = [256, 512, 1024, 2048]

# Times in milliseconds
naive_times = [0.25, 2.01, 15.90, 75.71]
tiled_times = [0.16, 1.23, 8.31, 49.81]
cublas_times = [0.06, 0.27, 0.82, 6.43]

# GFLOPS
naive_gflops = [135.04, 133.67, 135.09, 226.91]
tiled_gflops = [207.11, 218.06, 258.33, 344.89]
cublas_gflops = [602.49, 997.31, 2616.21, 2669.89]

# Chart 1: Execution Time Comparison (Log Scale)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(sizes))
width = 0.25

bars1 = ax1.bar(x - width, naive_times, width, label='Naive', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x, tiled_times, width, label='Tiled (TILE=16)', color='#3498db', alpha=0.8)
bars3 = ax1.bar(x + width, cublas_times, width, label='cuBLAS', color='#2ecc71', alpha=0.8)

ax1.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Matrix Size', fontsize=12, fontweight='bold')
ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)
ax1.legend()
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar in bars1:
    ax1.annotate(f'{bar.get_height():.2f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

# Chart 2: GFLOPS Comparison
bars1 = ax2.bar(x - width, naive_gflops, width, label='Naive', color='#e74c3c', alpha=0.8)
bars2 = ax2.bar(x, tiled_gflops, width, label='Tiled (TILE=16)', color='#3498db', alpha=0.8)
bars3 = ax2.bar(x + width, cublas_gflops, width, label='cuBLAS', color='#2ecc71', alpha=0.8)

ax2.set_ylabel('GFLOPS', fontsize=12, fontweight='bold')
ax2.set_xlabel('Matrix Size', fontsize=12, fontweight='bold')
ax2.set_title('Performance (GFLOPS - Higher is Better)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(sizes)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add peak performance line
peak_gflops = 4800  # GTX 1650 theoretical peak
ax2.axhline(y=peak_gflops, color='gray', linestyle='--', alpha=0.5, label=f'Theoretical Peak ({peak_gflops} GFLOPS)')

plt.tight_layout()
plt.savefig('matmul_benchmark_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: matmul_benchmark_comparison.png")

# Chart 3: Speedup Chart
fig, ax = plt.subplots(figsize=(10, 5))

# Calculate speedups
tiled_vs_naive = [n/t for n, t in zip(naive_times, tiled_times)]
cublas_vs_naive = [n/c for n, c in zip(naive_times, cublas_times)]
cublas_vs_tiled = [t/c for t, c in zip(tiled_times, cublas_times)]

width = 0.25
x = np.arange(len(sizes))

bars1 = ax.bar(x - width, tiled_vs_naive, width, label='Tiled vs Naive', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, cublas_vs_tiled, width, label='cuBLAS vs Tiled', color='#2ecc71', alpha=0.8)
bars3 = ax.bar(x + width, cublas_vs_naive, width, label='cuBLAS vs Naive', color='#9b59b6', alpha=0.8)

ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
ax.set_xlabel('Matrix Size', fontsize=12, fontweight='bold')
ax.set_title('Speedup Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax.annotate(f'{bar.get_height():.1f}×',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('matmul_speedup_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: matmul_speedup_comparison.png")

# Chart 4: Memory Bandwidth Utilization
fig, ax = plt.subplots(figsize=(10, 5))

# Theoretical bandwidth: 160 GB/s for GTX 1650 Max-Q
peak_bandwidth = 160  # GB/s

# Calculate effective bandwidth for each implementation
# For naive: each element reads M+N floats = 2*N floats for square matrix
# For tiled: reads are reduced by TILE_SIZE factor
# Approximate calculation based on timing and data movement

# For 1024x1024:
# Naive reads: 1024*1024 * 2048 * 4 bytes = 8 GB of reads (per thread model, but coalesced)
# Actually, let's show GFLOPS efficiency instead

efficiencies = []
for ng, tg, cg in zip(naive_gflops, tiled_gflops, cublas_gflops):
    efficiencies.append([ng/peak_gflops*100, tg/peak_gflops*100, cg/peak_gflops*100])

efficiencies = np.array(efficiencies)

ax.plot(sizes, efficiencies[:, 0], 'o-', label='Naive', color='#e74c3c', linewidth=2, markersize=8)
ax.plot(sizes, efficiencies[:, 1], 's-', label='Tiled', color='#3498db', linewidth=2, markersize=8)
ax.plot(sizes, efficiencies[:, 2], '^-', label='cuBLAS', color='#2ecc71', linewidth=2, markersize=8)

ax.set_ylabel('Efficiency (% of Peak GFLOPS)', fontsize=12, fontweight='bold')
ax.set_xlabel('Matrix Size', fontsize=12, fontweight='bold')
ax.set_title('GPU Efficiency (GTX 1650 Max-Q Peak: 4.8 TFLOPS)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Add percentage labels
for i, (x_pos, eff) in enumerate(zip(sizes, efficiencies)):
    ax.annotate(f'{eff[2]:.0f}%', 
                xy=(i, eff[2]), 
                xytext=(5, 5), textcoords="offset points",
                fontsize=9, color='#2ecc71')

plt.tight_layout()
plt.savefig('matmul_efficiency.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: matmul_efficiency.png")

print("\n✅ All charts generated successfully!")
print("\nFiles created:")
print("  - matmul_benchmark_comparison.png")
print("  - matmul_speedup_comparison.png")
print("  - matmul_efficiency.png")
