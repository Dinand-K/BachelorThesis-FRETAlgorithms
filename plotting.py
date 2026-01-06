import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import os

# --- 1. SETUP & CONFIG PARSING ---

# Snakemake Inputs/Params
class_files   = snakemake.input.stats_class
global_files  = snakemake.input.stats_global
alg_names     = snakemake.params.alg_names
sim_config    = snakemake.params.sim_settings

out_counts    = snakemake.output.counts_plots
out_retention = snakemake.output.retention_plots
out_pr        = snakemake.output.pr_plot
out_total     = snakemake.output.total_visualization

# HARDCODED TARGET FOR NOW (As requested)
# This matches the behavior of matching_and_benchmarking.py which defaults to Stationary = Signal
TARGET_DISPLAY_NAME = "Stationary" 

# Verify if user changed the display name in config, if so, update our target name
# This ensures that if you renamed "Stationary" to "Fixed" in config, plotting still works.
if "stationary" in sim_config:
    TARGET_DISPLAY_NAME = sim_config["stationary"].get("display_name", "Stationary")

# --- GLOBAL DEFINITIONS ---
COLOR_CORRECT_RAW   = '#a1d99b'  # Lighter green
COLOR_CORRECT_FILT  = '#41ab5d'  # Darker green
COLOR_ABERRANT_RAW  = '#fc9272'  # Lighter red
COLOR_ABERRANT_FILT = '#cb181d'  # Darker red

# --- HELPER FUNCTIONS ---

def get_yerr(df, col_top, col_sub_mean, col_sub_p05, col_sub_p95):
    """Calculates asymmetric error bars (lower, upper) for Matplotlib."""
    means = df[col_top][col_sub_mean]
    p05 = df[col_top][col_sub_p05]
    p95 = df[col_top][col_sub_p95]
    lower_err = (means - p05).clip(lower=0)
    upper_err = (p95 - means).clip(lower=0)
    return [lower_err, upper_err]

def prepare_data(df_input):
    """Cleans and reorders dataframe (Target Class first)."""
    df_clean = df_input.copy()
    
    # Remove index name row if present (pandas artifact sometimes)
    if df_clean.iloc[0, 0] == 'Class':
        df_clean = df_clean.drop(0)
    
    # Sort: Target Class first, others follow
    target_mask = df_clean.iloc[:, 0] == TARGET_DISPLAY_NAME
    
    if target_mask.sum() > 0:
        df_final = pd.concat([df_clean[target_mask], df_clean[~target_mask]]).reset_index(drop=True)
    else:
        df_final = df_clean
        
    # Ensure numeric types
    cols = ['Raw_Count', 'Filtered_Count', 'Retention']
    for c in cols:
        if c in df_final.columns:
            for sub in ['mean', 'p05', 'p95']:
                if (c, sub) in df_final.columns:
                     df_final[(c, sub)] = pd.to_numeric(df_final[(c, sub)])
    return df_final

# --- PLOTTING FUNCTIONS ---

def plot_counts(df, alg_name, ax, show_legend=False):
    """Plots Raw vs Filtered Counts."""
    classes = df.iloc[:, 0]
    x = np.arange(len(classes))
    width = 0.35
    
    raw_means = df['Raw_Count']['mean']
    raw_err = get_yerr(df, 'Raw_Count', 'mean', 'p05', 'p95')
    filt_means = df['Filtered_Count']['mean']
    filt_err = get_yerr(df, 'Filtered_Count', 'mean', 'p05', 'p95')

    # Color Logic: Green if Target, Red otherwise
    raw_colors = [COLOR_CORRECT_RAW if c == TARGET_DISPLAY_NAME else COLOR_ABERRANT_RAW for c in classes]
    filt_colors = [COLOR_CORRECT_FILT if c == TARGET_DISPLAY_NAME else COLOR_ABERRANT_FILT for c in classes]

    ax.bar(x - width/2, raw_means, width, yerr=raw_err, capsize=4, 
           color=raw_colors, error_kw=dict(lw=1, capthick=1))
    ax.bar(x + width/2, filt_means, width, yerr=filt_err, capsize=4, 
           color=filt_colors, error_kw=dict(lw=1, capthick=1))

    ax.set_ylabel('Count')
    ax.set_title(f'{alg_name}\nDetections')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha='right')
    
    if show_legend:
        legend_elements = [
            Patch(facecolor=COLOR_CORRECT_FILT, label='Correct Data'),
            Patch(facecolor=COLOR_ABERRANT_FILT, label='Aberrant Data'),
        ]
        ax.legend(handles=legend_elements, fontsize='small', loc='upper right')

def plot_retention(df, alg_name, ax, show_legend=False):
    """Plots Retention Rate."""
    classes = df.iloc[:, 0]
    x = np.arange(len(classes))
    width = 0.6
    
    means = df['Retention']['mean']
    err = get_yerr(df, 'Retention', 'mean', 'p05', 'p95')
    
    # Color Logic
    colors = [COLOR_CORRECT_FILT if c == TARGET_DISPLAY_NAME else COLOR_ABERRANT_FILT for c in classes]

    rects = ax.bar(x, means, width, yerr=err, capsize=5, color=colors)

    ax.set_ylabel('Retention (0-1)')
    ax.set_title(f'{alg_name}\nRetention')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha='right')
    
    y_max = (means + err[1]).max()
    if pd.isna(y_max): y_max = 1.0
    ax.set_ylim(0, max(1.05, y_max * 1.1))

    for rect in rects:
        h = rect.get_height()
        if not np.isnan(h):
            ax.annotate(f'{h:.0%}', xy=(rect.get_x()+rect.get_width()/2, h),
                        xytext=(0, 5), textcoords="offset points", ha='center', fontsize=8)
            
    if show_legend:
        legend_elements = [
            Patch(facecolor=COLOR_CORRECT_FILT, label='Correct Data'),
            Patch(facecolor=COLOR_ABERRANT_FILT, label='Aberrant Data'),
        ]
        ax.legend(handles=legend_elements, fontsize='small', loc='upper right')

def plot_global_pr(all_global_dfs, alg_names, ax):
    """Plots Precision vs Recall for ALL algorithms with X and Y error bars."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(alg_names)))
    legend_handles = []
    
    for i, (df, name) in enumerate(zip(all_global_dfs, alg_names)):
        c = colors[i]
        
        # Legend handle
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=name, 
                                     markerfacecolor=c, markersize=10))

        # --- Helper to extract values ---
        def get_val(metric, level1, level2):
            row = df[df.iloc[:, 0] == metric]
            if row.empty: return None
            return float(row[level1][level2].values[0])

        # --- Helper to get error ranges ---
        def get_stats_tuple(metric, level1):
            mean_v = get_val(metric, level1, 'mean')
            if mean_v is None: return None, None
            p05 = get_val(metric, level1, 'p05')
            p95 = get_val(metric, level1, 'p95')
            return mean_v, [[max(0, mean_v - p05)], [max(0, p95 - mean_v)]]

        # --- Helper to calculate Recall ---
        def calc_rec(p, f1): 
            return (f1 * p) / (2 * p - f1) if (2*p - f1) != 0 else 0

        # 1. Get Precision (Y-axis)
        raw_prec, raw_yerr = get_stats_tuple('Global_Precision', 'Raw')
        filt_prec, filt_yerr = get_stats_tuple('Global_Precision', 'Filtered')

        # 2. Get Recall (X-axis) - ALWAYS calculate from F1 & Precision
        raw_f1, _ = get_stats_tuple('Global_F1_Score', 'Raw')
        filt_f1, _ = get_stats_tuple('Global_F1_Score', 'Filtered')
        
        # Means
        raw_rec = calc_rec(raw_prec, raw_f1)
        filt_rec = calc_rec(filt_prec, filt_f1)
        
        # Errors (Calculated)
        def get_bound_err(level, mean_rec):
            f1_p05 = get_val('Global_F1_Score', level, 'p05')
            f1_p95 = get_val('Global_F1_Score', level, 'p95')
            p_p05 = get_val('Global_Precision', level, 'p05')
            p_p95 = get_val('Global_Precision', level, 'p95')
            
            # Approximate worst-case recall bounds
            rec_p05 = calc_rec(p_p05, f1_p05)
            rec_p95 = calc_rec(p_p95, f1_p95)
            
            return [[max(0, mean_rec - rec_p05)], [max(0, rec_p95 - mean_rec)]]

        raw_xerr = get_bound_err('Raw', raw_rec)
        filt_xerr = get_bound_err('Filtered', filt_rec)

        if any(v is None for v in [raw_prec, raw_rec, filt_prec, filt_rec]):
            continue

        # Plotting
        ax.errorbar(raw_rec, raw_prec, xerr=raw_xerr, yerr=raw_yerr, 
                    fmt='none', ecolor=c, alpha=0.5, capsize=3)
        ax.scatter(raw_rec, raw_prec, color=c, marker='o', facecolors='none', s=80, zorder=2)
        
        ax.errorbar(filt_rec, filt_prec, xerr=filt_xerr, yerr=filt_yerr, 
                    fmt='none', ecolor=c, capsize=3)
        ax.scatter(filt_rec, filt_prec, color=c, marker='o', s=80, zorder=2)
        
        ax.annotate("", xy=(filt_rec, filt_prec), xytext=(raw_rec, raw_prec),
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.5, ls='--'), zorder=1)
        
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Global Performance')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.legend(handles=legend_handles, loc='lower left', fontsize='small', title="Algorithm")


# --- MAIN EXECUTION ---

# Load Data
dfs_class = [prepare_data(pd.read_csv(f, header=[0,1])) for f in class_files]
dfs_global = [pd.read_csv(f, header=[0,1]) for f in global_files]
num_algs = len(alg_names)

# 1. Generate Individual Plots
for i, name in enumerate(alg_names):
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_counts(dfs_class[i], name, ax, show_legend=True)
    plt.tight_layout(); plt.savefig(out_counts[i], dpi=300); plt.close()
    
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_retention(dfs_class[i], name, ax, show_legend=True)
    plt.tight_layout(); plt.savefig(out_retention[i], dpi=300); plt.close()

# Global Plot
fig, ax = plt.subplots(figsize=(6, 6))
plot_global_pr(dfs_global, alg_names, ax)
plt.tight_layout(); plt.savefig(out_pr, dpi=300); plt.close()


# 2. Generate TOTAL VISUALIZATION (Master Panel)
fig_height = 10
# Calculate width dynamically based on number of algorithms
fig_width = (4 * num_algs) + (fig_height * 0.8) 

fig = plt.figure(figsize=(fig_width, fig_height))

ws_ratios = [1] * num_algs + [2.5] 
gs = gridspec.GridSpec(2, num_algs + 1, width_ratios=ws_ratios, hspace=0.3, wspace=0.3)

# Row 1: Counts
for i in range(num_algs):
    ax = fig.add_subplot(gs[0, i])
    show_leg = (i == num_algs - 1)
    plot_counts(dfs_class[i], alg_names[i], ax, show_legend=show_leg)
    ax.text(-0.15, 1.05, chr(65+i), transform=ax.transAxes, size=16, weight='bold')

# Row 2: Retention
for i in range(num_algs):
    ax = fig.add_subplot(gs[1, i])
    show_leg = (i == num_algs - 1)
    plot_retention(dfs_class[i], alg_names[i], ax, show_legend=show_leg)
    ax.text(-0.15, 1.05, chr(65+num_algs+i), transform=ax.transAxes, size=16, weight='bold')

# Right Column: Global PR
ax_global = fig.add_subplot(gs[:, -1])
plot_global_pr(dfs_global, alg_names, ax_global)
ax_global.text(-0.1, 1.02, chr(65 + 2*num_algs), transform=ax_global.transAxes, size=16, weight='bold')

plt.savefig(out_total, dpi=300, bbox_inches='tight')
plt.close()