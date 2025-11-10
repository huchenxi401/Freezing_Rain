import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

file_path = "../../../data/freezing_rain_stratified_sample_414_counties.xlsx"
df = pd.read_excel(file_path)

ice_thickness_llm = df['ICE_THICKNESS_INCHES']
ice_thickness_human = df['ICE_THICKNESS_INCHES_CHECK']
damage_severity_llm = df['DAMAGE_SEVERITY']
damage_severity_human = df['DAMAGE_SEVERITY_CHECK']

plt.figure(figsize=(6, 6))

valid_mask = pd.notna(ice_thickness_llm) & pd.notna(ice_thickness_human)
x_valid = ice_thickness_human[valid_mask]
y_valid = ice_thickness_llm[valid_mask]

plt.scatter(x_valid, y_valid, alpha=0.6, s=80, color='steelblue', edgecolors='white', linewidth=0.5)

if len(x_valid) > 0 and len(y_valid) > 0:
    min_val = min(min(x_valid), min(y_valid))
    max_val = max(max(x_valid), max(y_valid))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Reference Line')

plt.xlabel('Manual Identified ICE Thickness (inches)', fontsize=14)
plt.ylabel('LLM Identified ICE Thickness (inches)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
if len(x_valid) > 1:
    correlation = np.corrcoef(x_valid, y_valid)[0, 1]
    r_squared = correlation ** 2
    plt.text(0.05, 0.86, f'R² = {r_squared:.2f}', transform=plt.gca().transAxes, 
             fontsize=14, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.axis('equal')

plt.tight_layout()
plt.savefig('./Ice_thickness_comparison_414county.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(6, 6))

severity_order = ['Low', 'Medium', 'High']

confusion_matrix = pd.crosstab(
    damage_severity_llm, 
    damage_severity_human, 
    rownames=['LLM Identified'], 
    colnames=['Human Identified']
)

for category in severity_order:
    if category not in confusion_matrix.index:
        confusion_matrix.loc[category] = 0
    if category not in confusion_matrix.columns:
        confusion_matrix[category] = 0

confusion_matrix = confusion_matrix.reindex(severity_order).reindex(columns=severity_order)
confusion_matrix = confusion_matrix.fillna(0).astype(int)

sns.heatmap(confusion_matrix, 
           annot=True, 
           fmt='d', 
           cmap='Blues',
           square=True,
           cbar_kws={'label': 'Count', 'shrink': 0.8},
           annot_kws={'size': 16, 'weight': 'bold'},
           linewidths=1,
           linecolor='white')


plt.ylabel('LLM Identified Damage Severity', fontsize=14)
plt.xlabel('Manual Identified Damage Severity', fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
if confusion_matrix.sum().sum() > 0:
    accuracy = np.trace(confusion_matrix) / confusion_matrix.sum().sum()
    plt.text(0.02, 0.98, f'Accuracy = {accuracy:.1%}', transform=plt.gca().transAxes, 
             fontsize=14, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             verticalalignment='top')
plt.tight_layout()
plt.savefig('./Damage_severity_confusion_matrix_414county.png', dpi=300, bbox_inches='tight')
