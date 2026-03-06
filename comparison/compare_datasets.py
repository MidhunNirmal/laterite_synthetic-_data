"""
Comprehensive Dataset Comparison - Laterite Data
Compares: KDE-Copula Synthetic, CTGAN Synthetic, and MissForest Imputed vs Original
"""
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("COMPREHENSIVE DATASET COMPARISON")
print("="*80)

print("\nLoading datasets...")
orig = pd.read_csv("../laterite.csv")
synth_kde = pd.read_csv("../kdecopula_laterite/my_synthetic.csv")
synth_ctgan = pd.read_csv("../ctgan_synthetic_500.csv")
imp = pd.read_csv("../imputed_missforest.csv")

print(f"Original shape: {orig.shape}")
print(f"KDE-Copula Synthetic shape: {synth_kde.shape}")
print(f"CTGAN Synthetic shape: {synth_ctgan.shape}")
print(f"Imputed shape: {imp.shape}")

# Get common numeric columns
orig_numeric = set(orig.select_dtypes(include=[np.number]).columns)
kde_numeric = set(synth_kde.select_dtypes(include=[np.number]).columns)
ctgan_numeric = set(synth_ctgan.select_dtypes(include=[np.number]).columns)
imp_numeric = set(imp.select_dtypes(include=[np.number]).columns)

common = sorted(list(orig_numeric & kde_numeric & ctgan_numeric & imp_numeric))
print(f"\nCommon columns ({len(common)}): {common}")

# Calculate metrics
print("\n" + "="*80)
print("DISTANCE METRICS")
print("="*80)

results = []
for col in common:
    o = orig[col].dropna().values
    s_kde = synth_kde[col].dropna().values
    s_ctgan = synth_ctgan[col].dropna().values
    i = imp[col].dropna().values
    
    if len(o) == 0:
        continue
    
    # Wasserstein
    w_kde = wasserstein_distance(o, s_kde)
    w_ctgan = wasserstein_distance(o, s_ctgan)
    w_i = wasserstein_distance(o, i)
    
    # KS test
    ks_kde = ks_2samp(o, s_kde).statistic
    ks_ctgan = ks_2samp(o, s_ctgan).statistic
    ks_i = ks_2samp(o, i).statistic
    
    # JS divergence
    bins = np.linspace(
        min(o.min(), s_kde.min(), s_ctgan.min(), i.min()),
        max(o.max(), s_kde.max(), s_ctgan.max(), i.max()),
        50
    )
    
    h_o, _ = np.histogram(o, bins=bins, density=True)
    h_kde, _ = np.histogram(s_kde, bins=bins, density=True)
    h_ctgan, _ = np.histogram(s_ctgan, bins=bins, density=True)
    h_i, _ = np.histogram(i, bins=bins, density=True)
    
    h_o = h_o / (h_o.sum() + 1e-10)
    h_kde = h_kde / (h_kde.sum() + 1e-10)
    h_ctgan = h_ctgan / (h_ctgan.sum() + 1e-10)
    h_i = h_i / (h_i.sum() + 1e-10)
    
    js_kde = jensenshannon(h_o, h_kde)
    js_ctgan = jensenshannon(h_o, h_ctgan)
    js_i = jensenshannon(h_o, h_i)
    
    results.append({
        'Column': col,
        'Wass_KDE': w_kde, 'Wass_CTGAN': w_ctgan, 'Wass_Imp': w_i,
        'KS_KDE': ks_kde, 'KS_CTGAN': ks_ctgan, 'KS_Imp': ks_i,
        'JS_KDE': js_kde, 'JS_CTGAN': js_ctgan, 'JS_Imp': js_i
    })
    
    # Determine winner for each metric
    wass_winner = min([('KDE', w_kde), ('CTGAN', w_ctgan), ('Imp', w_i)], key=lambda x: x[1])[0]
    ks_winner = min([('KDE', ks_kde), ('CTGAN', ks_ctgan), ('Imp', ks_i)], key=lambda x: x[1])[0]
    js_winner = min([('KDE', js_kde), ('CTGAN', js_ctgan), ('Imp', js_i)], key=lambda x: x[1])[0]
    
    print(f"\n{col}:")
    print(f"  Wasserstein: KDE={w_kde:.3f}, CTGAN={w_ctgan:.3f}, Imp={w_i:.3f} → {wass_winner} wins")
    print(f"  KS Stat:     KDE={ks_kde:.3f}, CTGAN={ks_ctgan:.3f}, Imp={ks_i:.3f} → {ks_winner} wins")
    print(f"  JS Div:      KDE={js_kde:.3f}, CTGAN={js_ctgan:.3f}, Imp={js_i:.3f} → {js_winner} wins")

df = pd.DataFrame(results)
df.to_csv('distance_metrics_full.csv', index=False)
print("\nSaved: distance_metrics_full.csv")

# Count wins
wins_kde = sum([
    (df['Wass_KDE'] < df['Wass_CTGAN']).sum() + (df['Wass_KDE'] < df['Wass_Imp']).sum(),
    (df['KS_KDE'] < df['KS_CTGAN']).sum() + (df['KS_KDE'] < df['KS_Imp']).sum(),
    (df['JS_KDE'] < df['JS_CTGAN']).sum() + (df['JS_KDE'] < df['JS_Imp']).sum()
])

wins_ctgan = sum([
    (df['Wass_CTGAN'] < df['Wass_KDE']).sum() + (df['Wass_CTGAN'] < df['Wass_Imp']).sum(),
    (df['KS_CTGAN'] < df['KS_KDE']).sum() + (df['KS_CTGAN'] < df['KS_Imp']).sum(),
    (df['JS_CTGAN'] < df['JS_KDE']).sum() + (df['JS_CTGAN'] < df['JS_Imp']).sum()
])

wins_imp = sum([
    (df['Wass_Imp'] < df['Wass_KDE']).sum() + (df['Wass_Imp'] < df['Wass_CTGAN']).sum(),
    (df['KS_Imp'] < df['KS_KDE']).sum() + (df['KS_Imp'] < df['KS_CTGAN']).sum(),
    (df['JS_Imp'] < df['JS_KDE']).sum() + (df['JS_Imp'] < df['JS_CTGAN']).sum()
])

# Generate report
report = f"""
{'='*80}
COMPREHENSIVE DATASET COMPARISON REPORT
{'='*80}

Datasets:
  - Original: laterite.csv ({orig.shape[0]} rows)
  - KDE-Copula Synthetic: my_synthetic.csv ({synth_kde.shape[0]} rows)
  - CTGAN Synthetic: ctgan_synthetic_500.csv ({synth_ctgan.shape[0]} rows)
  - MissForest Imputed: imputed_missforest.csv ({imp.shape[0]} rows)

Common columns: {len(common)}

RESULTS (Lower is better):
  Wasserstein Distance:
    - KDE-Copula wins: {(df['Wass_KDE'] <= df[['Wass_CTGAN', 'Wass_Imp']].min(axis=1)).sum()}/{len(common)}
    - CTGAN wins: {(df['Wass_CTGAN'] <= df[['Wass_KDE', 'Wass_Imp']].min(axis=1)).sum()}/{len(common)}
    - Imputed wins: {(df['Wass_Imp'] <= df[['Wass_KDE', 'Wass_CTGAN']].min(axis=1)).sum()}/{len(common)}
  
  KS Statistic:
    - KDE-Copula wins: {(df['KS_KDE'] <= df[['KS_CTGAN', 'KS_Imp']].min(axis=1)).sum()}/{len(common)}
    - CTGAN wins: {(df['KS_CTGAN'] <= df[['KS_KDE', 'KS_Imp']].min(axis=1)).sum()}/{len(common)}
    - Imputed wins: {(df['KS_Imp'] <= df[['KS_KDE', 'KS_CTGAN']].min(axis=1)).sum()}/{len(common)}
  
  JS Divergence:
    - KDE-Copula wins: {(df['JS_KDE'] <= df[['JS_CTGAN', 'JS_Imp']].min(axis=1)).sum()}/{len(common)}
    - CTGAN wins: {(df['JS_CTGAN'] <= df[['JS_KDE', 'JS_Imp']].min(axis=1)).sum()}/{len(common)}
    - Imputed wins: {(df['JS_Imp'] <= df[['JS_KDE', 'JS_CTGAN']].min(axis=1)).sum()}/{len(common)}

OVERALL RANKING:
  1st: {'Imputed' if wins_imp >= max(wins_kde, wins_ctgan) else 'CTGAN' if wins_ctgan >= wins_kde else 'KDE-Copula'}
  2nd: {'CTGAN' if wins_imp >= max(wins_kde, wins_ctgan) and wins_ctgan >= wins_kde else 'KDE-Copula' if wins_imp >= max(wins_kde, wins_ctgan) else 'Imputed' if wins_ctgan >= wins_kde else 'CTGAN'}
  3rd: {'KDE-Copula' if wins_imp >= max(wins_kde, wins_ctgan) else 'Imputed' if wins_ctgan >= wins_kde else 'CTGAN'}

{'='*80}
"""

print(report)
with open('comparison_report_full.txt', 'w') as f:
    f.write(report)
print("Saved: comparison_report_full.txt")

# Create plots
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

n = len(common)
fig, axes = plt.subplots((n+2)//3, 3, figsize=(18, 5*((n+2)//3)))
axes = axes.flatten() if n > 1 else [axes]

for idx, col in enumerate(common):
    ax = axes[idx]
    ax.hist(orig[col].dropna(), bins=20, alpha=0.4, label='Original', density=True, color='black')
    ax.hist(synth_kde[col].dropna(), bins=20, alpha=0.4, label='KDE-Copula', density=True, color='blue')
    ax.hist(synth_ctgan[col].dropna(), bins=20, alpha=0.4, label='CTGAN', density=True, color='green')
    ax.hist(imp[col].dropna(), bins=20, alpha=0.4, label='Imputed', density=True, color='red')
    ax.set_title(col, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

for idx in range(n, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('distributions_full.png', dpi=250)
print("Saved: distributions_full.png")
plt.close()

# Boxplot comparison
fig, axes = plt.subplots((n+2)//3, 3, figsize=(18, 5*((n+2)//3)))
axes = axes.flatten() if n > 1 else [axes]

for idx, col in enumerate(common):
    ax = axes[idx]
    
    data_to_plot = [
        orig[col].dropna(),
        synth_kde[col].dropna(),
        synth_ctgan[col].dropna(),
        imp[col].dropna()
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['Original', 'KDE', 'CTGAN', 'Imputed'],
                   patch_artist=True, showmeans=True)
    
    colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title(col, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

for idx in range(n, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('boxplot_full.png', dpi=250)
print("Saved: boxplot_full.png")
plt.close()

# Q-Q Plot comparison
print("\nGenerating Q-Q plots...")
fig, axes = plt.subplots((n+2)//3, 3, figsize=(18, 5*((n+2)//3)))
axes = axes.flatten() if n > 1 else [axes]

for idx, col in enumerate(common):
    ax = axes[idx]
    
    # Get data
    o_data = np.sort(orig[col].dropna().values)
    kde_data = np.sort(synth_kde[col].dropna().values)
    ctgan_data = np.sort(synth_ctgan[col].dropna().values)
    imp_data = np.sort(imp[col].dropna().values)
    
    # Interpolate to same length for Q-Q plot
    n_quantiles = min(len(o_data), 100)
    quantiles = np.linspace(0, 100, n_quantiles)
    
    o_q = np.percentile(o_data, quantiles)
    kde_q = np.percentile(kde_data, quantiles)
    ctgan_q = np.percentile(ctgan_data, quantiles)
    imp_q = np.percentile(imp_data, quantiles)
    
    # Plot Q-Q
    ax.scatter(o_q, kde_q, alpha=0.5, s=20, label='KDE-Copula', color='blue')
    ax.scatter(o_q, ctgan_q, alpha=0.5, s=20, label='CTGAN', color='green')
    ax.scatter(o_q, imp_q, alpha=0.5, s=20, label='Imputed', color='red')
    
    # Add diagonal reference line
    min_val = min(o_q.min(), kde_q.min(), ctgan_q.min(), imp_q.min())
    max_val = max(o_q.max(), kde_q.max(), ctgan_q.max(), imp_q.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Original Quantiles', fontsize=9)
    ax.set_ylabel('Comparison Quantiles', fontsize=9)
    ax.set_title(col, fontweight='bold')
    ax.legend(fontsize=7, loc='best')
    ax.grid(alpha=0.3)

for idx in range(n, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('qq_plots_full.png', dpi=250)
print("Saved: qq_plots_full.png")
plt.close()

# Correlation Matrix comparison
print("\nGenerating Correlation plots...")
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

datasets = [orig[common], synth_kde[common], synth_ctgan[common], imp[common]]
names = ['Original', 'KDE-Copula', 'CTGAN', 'Imputed']

# Calculate correlation matrices
corrs = [df.corr() for df in datasets]

# Get global min/max for consisten color scale
vmin = min([c.min().min() for c in corrs])
vmax = max([c.max().max() for c in corrs])

for i, (corr, name) in enumerate(zip(corrs, names)):
    sns.heatmap(corr, ax=axes[i], vmin=vmin, vmax=vmax, cmap='coolwarm', 
                cbar=i==3, square=True, annot=False)
    axes[i].set_title(f"{name} Correlation", fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_full.png', dpi=250)
print("Saved: correlation_full.png")
plt.close()

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - distributions_full.png")
print("  - boxplot_full.png")
print("  - qq_plots_full.png")
print("  - correlation_full.png")
print("  - distance_metrics_full.csv")
print("  - comparison_report_full.txt")
