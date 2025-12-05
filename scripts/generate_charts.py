#!/usr/bin/env python3
"""
Generate comparison charts for PowerElecBench evaluation results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Results data from evaluations
results = {
    'Grok 4.1 Fast': {
        'L1': (92, 150, 61.3),
        'L2': (96, 150, 64.0),
        'L3': (54, 100, 54.0),
        'L4': (21, 100, 21.0),
        'total': (263, 500, 52.6),
        'cost': 0.12,
        'latency': 4.0,
        'color': '#FF6B6B'
    },
    'GPT-4o': {
        'L1': (91, 150, 60.7),
        'L2': (88, 150, 58.7),
        'L3': (57, 100, 57.0),
        'L4': (9, 100, 9.0),
        'total': (245, 500, 49.0),
        'cost': 3.00,
        'latency': 12.0,
        'color': '#4ECDC4'
    }
}

# Create output directory
os.makedirs('benchmarks/charts', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# ============================================================
# Chart 1: Accuracy by Level (Grouped Bar Chart)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

levels = ['L1\n(150)', 'L2\n(150)', 'L3\n(100)', 'L4\n(100)', 'Total\n(500)']
x = np.arange(len(levels))
width = 0.35

grok_acc = [results['Grok 4.1 Fast']['L1'][2], results['Grok 4.1 Fast']['L2'][2], 
            results['Grok 4.1 Fast']['L3'][2], results['Grok 4.1 Fast']['L4'][2],
            results['Grok 4.1 Fast']['total'][2]]
gpt_acc = [results['GPT-4o']['L1'][2], results['GPT-4o']['L2'][2],
           results['GPT-4o']['L3'][2], results['GPT-4o']['L4'][2],
           results['GPT-4o']['total'][2]]

bars1 = ax.bar(x - width/2, grok_acc, width, label='Grok 4.1 Fast', color='#FF6B6B', edgecolor='black')
bars2 = ax.bar(x + width/2, gpt_acc, width, label='GPT-4o', color='#4ECDC4', edgecolor='black')

ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_xlabel('Difficulty Level', fontsize=14)
ax.set_title('PowerElecBench: Model Accuracy by Difficulty Level\n(Vout within 5% error)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(levels)
ax.legend(loc='upper right', fontsize=12)
ax.set_ylim(0, 80)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('benchmarks/charts/accuracy_by_level.png', dpi=300, bbox_inches='tight')
plt.savefig('benchmarks/charts/accuracy_by_level.pdf', bbox_inches='tight')
print("✅ Saved: accuracy_by_level.png/pdf")

# ============================================================
# Chart 2: Cost vs Accuracy Scatter
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

models = list(results.keys())
costs = [results[m]['cost'] for m in models]
accuracies = [results[m]['total'][2] for m in models]
colors = [results[m]['color'] for m in models]

scatter = ax.scatter(costs, accuracies, s=500, c=colors, edgecolors='black', linewidths=2, alpha=0.8)

for i, model in enumerate(models):
    ax.annotate(model, (costs[i], accuracies[i]), textcoords="offset points",
                xytext=(0, 15), ha='center', fontsize=12, fontweight='bold')

ax.set_xlabel('Cost for 500 Problems ($)', fontsize=14)
ax.set_ylabel('Overall Accuracy (%)', fontsize=14)
ax.set_title('PowerElecBench: Cost vs Accuracy Trade-off', fontsize=16, fontweight='bold')
ax.set_xlim(-0.5, 4)
ax.set_ylim(40, 60)

# Add efficiency annotation
ax.annotate('Better\n(Lower cost, Higher accuracy)', xy=(0.3, 55), fontsize=10, 
            color='green', ha='center', style='italic')

plt.tight_layout()
plt.savefig('benchmarks/charts/cost_vs_accuracy.png', dpi=300, bbox_inches='tight')
plt.savefig('benchmarks/charts/cost_vs_accuracy.pdf', bbox_inches='tight')
print("✅ Saved: cost_vs_accuracy.png/pdf")

# ============================================================
# Chart 3: Performance Summary Table as Figure
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

table_data = [
    ['Model', 'L1 (150)', 'L2 (150)', 'L3 (100)', 'L4 (100)', 'Total', 'Cost', 'Latency'],
    ['Grok 4.1 Fast', '92 (61.3%)', '96 (64.0%)', '54 (54.0%)', '21 (21.0%)', '263 (52.6%)', '$0.12', '~4s'],
    ['GPT-4o', '91 (60.7%)', '88 (58.7%)', '57 (57.0%)', '9 (9.0%)', '245 (49.0%)', '$3.00', '~12s'],
]

colors_table = [['#E8E8E8']*8,
                ['#FFE5E5']*8,
                ['#E5F5F3']*8]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                 cellColours=colors_table, colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.12, 0.08, 0.08])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style header row
for j in range(8):
    table[(0, j)].set_text_props(fontweight='bold')

ax.set_title('PowerElecBench Evaluation Results Summary', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('benchmarks/charts/results_table.png', dpi=300, bbox_inches='tight')
plt.savefig('benchmarks/charts/results_table.pdf', bbox_inches='tight')
print("✅ Saved: results_table.png/pdf")

# ============================================================
# Chart 4: Radar Chart - Model Capabilities
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

categories = ['L1 Accuracy', 'L2 Accuracy', 'L3 Accuracy', 'L4 Accuracy', 'Cost Efficiency', 'Speed']
N = len(categories)

# Normalize metrics to 0-100 scale
grok_values = [61.3, 64.0, 54.0, 21.0, 100, 100]  # Cost/speed: 100 = best
gpt_values = [60.7, 58.7, 57.0, 9.0, 4, 33]  # Cost: 0.12/3.00 = 4%, Speed: 4/12 = 33%

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the loop

grok_values += grok_values[:1]
gpt_values += gpt_values[:1]

ax.plot(angles, grok_values, 'o-', linewidth=2, label='Grok 4.1 Fast', color='#FF6B6B')
ax.fill(angles, grok_values, alpha=0.25, color='#FF6B6B')
ax.plot(angles, gpt_values, 'o-', linewidth=2, label='GPT-4o', color='#4ECDC4')
ax.fill(angles, gpt_values, alpha=0.25, color='#4ECDC4')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 100)
ax.set_title('PowerElecBench: Model Capability Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

plt.tight_layout()
plt.savefig('benchmarks/charts/radar_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('benchmarks/charts/radar_comparison.pdf', bbox_inches='tight')
print("✅ Saved: radar_comparison.png/pdf")

# ============================================================
# Chart 5: Difficulty Progression Line Chart
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

levels_num = [1, 2, 3, 4]
grok_line = [61.3, 64.0, 54.0, 21.0]
gpt_line = [60.7, 58.7, 57.0, 9.0]

ax.plot(levels_num, grok_line, 'o-', linewidth=3, markersize=12, label='Grok 4.1 Fast', color='#FF6B6B')
ax.plot(levels_num, gpt_line, 's-', linewidth=3, markersize=12, label='GPT-4o', color='#4ECDC4')

ax.fill_between(levels_num, grok_line, alpha=0.2, color='#FF6B6B')
ax.fill_between(levels_num, gpt_line, alpha=0.2, color='#4ECDC4')

ax.set_xlabel('Difficulty Level', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('PowerElecBench: Accuracy Degradation with Difficulty', fontsize=16, fontweight='bold')
ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['L1\nBasic', 'L2\nIntermediate', 'L3\nAdvanced', 'L4\nExpert'])
ax.legend(loc='upper right', fontsize=12)
ax.set_ylim(0, 80)
ax.grid(True, alpha=0.3)

# Add annotations
ax.annotate('Both models\nstruggle at L4', xy=(4, 15), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('benchmarks/charts/difficulty_progression.png', dpi=300, bbox_inches='tight')
plt.savefig('benchmarks/charts/difficulty_progression.pdf', bbox_inches='tight')
print("✅ Saved: difficulty_progression.png/pdf")

print("\n" + "="*60)
print("📊 All charts saved to: benchmarks/charts/")
print("="*60)
