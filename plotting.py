import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import os

# --- 1. SETUP ---
class_files   = snakemake.input.stats_class
global_files  = snakemake.input.stats_global
alg_names     = snakemake.params.alg_names
sim_config    = snakemake.params.sim_settings
RealData      = snakemake.params.REALDATA
out_counts    = snakemake.output.counts_plots
out_retention = snakemake.output.retention_plots
out_pr        = snakemake.output.pr_plot
out_total     = snakemake.output.total_visualization
out_table     = snakemake.output.metrics_table

# Config parsing
TARGET_DISPLAY_NAME = "Stationary" 
if "stationary" in sim_config:
    TARGET_DISPLAY_NAME = sim_config["stationary"].get("display_name", "Stationary")

# --- COLORS ---
OKABE_ITO = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#E69F00', '#F0E442', '#56B4E9']
if len(alg_names) > len(OKABE_ITO):
    _colors = plt.cm.tab10(np.linspace(0, 1, len(alg_names)))
else:
    _colors = OKABE_ITO[:len(alg_names)]
ALG_COLOR_MAP = dict(zip(alg_names, _colors))

# --- VISUAL STYLE UPDATES ---
plt.rcParams.update({
    'font.family': 'serif', 
    'font.serif': ['Times New Roman'], 
    'font.size': 24,           
    'axes.titlesize': 48,      
    'axes.labelsize': 40,      
    'xtick.labelsize': 32,     
    'ytick.labelsize': 32,     
    'legend.fontsize': 28,     
    'figure.titlesize': 52    
})

# --- 2. HELPERS ---
def prepare_data(df_input):
    df = df_input.copy()
    if df.iloc[0, 0] == 'Class': df = df.drop(0)
    mask = df.iloc[:, 0] == TARGET_DISPLAY_NAME
    if mask.sum() > 0: df = pd.concat([df[mask], df[~mask]]).reset_index(drop=True)
    for c in ['Raw_Count', 'Filtered_Count', 'Retention']:
        if c in df.columns:
            for sub in ['mean', 'p025', 'p975']: 
                if (c, sub) in df.columns: df[(c, sub)] = pd.to_numeric(df[(c, sub)])
    return df

def organize_data_by_class(dfs_class, alg_names):
    all_classes = []
    seen = set()
    for df in dfs_class:
        for c in df.iloc[:, 0]:
            if c not in seen: all_classes.append(c); seen.add(c)
    class_map = {c: {} for c in all_classes}
    for df, alg in zip(dfs_class, alg_names):
        for _, row in df.iterrows():
            c_name = row.iloc[0]
            if c_name in class_map: class_map[c_name][alg] = row
    return all_classes, class_map

# --- 3. PLOTTING ---

def plot_overlapping_counts(class_name, class_data, alg_names, ax, show_ylabel=True, suppress_title=False):
    x = np.arange(len(alg_names))
    width = 0.6
    
    raw_means, filt_means = [], []
    # Structure for error bars: [[lower_errors], [upper_errors]]
    raw_errs, filt_errs = [[], []], [[], []]
    
    for alg in alg_names:
        if alg in class_data:
            row = class_data[alg]
            r_m = row['Raw_Count']['mean']
            f_m = row['Filtered_Count']['mean']
            
            raw_means.append(r_m)
            filt_means.append(f_m)
            
            if not RealData:
                r_lo, r_hi = row['Raw_Count']['p025'], row['Raw_Count']['p975']
                f_lo, f_hi = row['Filtered_Count']['p025'], row['Filtered_Count']['p975']
                
                # --- CALCULATION AND CLIPPING ---
                raw_low_len = max(0, r_m - r_lo) * 2
                raw_high_len = max(0, r_hi - r_m) * 2
                
                filt_low_len = max(0, f_m - f_lo) * 2
                filt_high_len = max(0, f_hi - f_m) * 2
                
                # Append Clipped Low, Standard High
                raw_errs[0].append(min(raw_low_len, r_m)) 
                raw_errs[1].append(raw_high_len)
                
                filt_errs[0].append(min(filt_low_len, f_m))
                filt_errs[1].append(filt_high_len)
        else:
            raw_means.append(0); filt_means.append(0)
            if not RealData: 
                for l in [raw_errs, filt_errs]: l[0].append(0); l[1].append(0)

    colors = [ALG_COLOR_MAP[a] for a in alg_names]
    
    # Establish Error Bars
    yerr_raw = raw_errs if not RealData else None
    yerr_filt = filt_errs if not RealData else None
    
    err_kw = dict(lw=2.5, capsize=6, capthick=2, ecolor='black', alpha=0.9)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # 1. Background (Detection) - Very light
    ax.bar(x, raw_means, width, yerr=yerr_raw, color=colors, 
           alpha=0.25, edgecolor=None, error_kw=err_kw, zorder=2, label='Detection')

    # 2. Foreground (Selection) - Semi-transparent
    ax.bar(x, filt_means, width, yerr=yerr_filt, color=colors, 
           alpha=0.6, edgecolor='black', linewidth=1, error_kw=err_kw, zorder=3, label='Selection')

    ax.set_xticks(x)
    ax.set_xticklabels(alg_names, rotation=30, ha='right', fontsize=32)
    
    ax.set_ylim(bottom=0)
    
    if show_ylabel: 
        ax.set_ylabel('Spot Count', fontweight='bold', labelpad=15)
    if not suppress_title: ax.set_title(f"{class_name}", fontweight='bold')


def plot_global_pr(dfs, names, ax):
    ax.set_aspect('equal', adjustable='box')
    X, Y = np.meshgrid(np.linspace(0.01, 1, 100), np.linspace(0.01, 1, 100))
    Z = 2 * X * Y / (X + Y)
    ax.clabel(ax.contour(X, Y, Z, levels=[0.2,0.4,0.6,0.7,0.8,0.9], 
              colors='lightgray', linestyles='--', alpha=0.6), inline=True, fmt='F1=%.1f', fontsize=22)
    
    markers = ['o', 's', '^', 'D', 'p', 'h', '*'][:len(names)]
    
    for i, (df, name) in enumerate(zip(dfs, names)):
        c, m = ALG_COLOR_MAP[name], markers[i]
        
        # Helper to extract mean and CI (if available)
        def get_stats(metric, lvl):
            row = df[df.iloc[:,0] == metric]
            if row.empty: return None, None, None
            
            mean_val = float(row[lvl]['mean'].values[0])
            
            if RealData:
                return mean_val, None, None
            else:
                lo_val = float(row[lvl]['p025'].values[0])
                hi_val = float(row[lvl]['p975'].values[0])
                return mean_val, lo_val, hi_val

        # 1. Get Values
        p_r, p_r_lo, p_r_hi       = get_stats('Global_Precision', 'Raw')
        p_f, p_f_lo, p_f_hi       = get_stats('Global_Precision', 'Filtered')
        rec_r, rec_r_lo, rec_r_hi = get_stats('Global_Recall', 'Raw')
        rec_f, rec_f_lo, rec_f_hi = get_stats('Global_Recall', 'Filtered')

        if None in [p_r, rec_r, p_f, rec_f]: continue

        # 2. Plot Error Bars (Background)
        # We plot these before markers so they appear behind (zorder=1.5 vs zorder=3)
        if not RealData:
            # Error Bar Formatting: Thin, light grey, no caps
            err_style = dict(fmt='none', ecolor="#323232", elinewidth=3, capsize=0, zorder=1.5)

            # Raw Errors (Asymmetric)
            # xerr shape: [[left_err], [right_err]]
            xerr_r = [[rec_r - rec_r_lo], [rec_r_hi - rec_r]]
            yerr_r = [[p_r - p_r_lo], [p_r_hi - p_r]]
            ax.errorbar(rec_r, p_r, xerr=xerr_r, yerr=yerr_r, **err_style)

            # Filtered Errors
            xerr_f = [[rec_f - rec_f_lo], [rec_f_hi - rec_f]]
            yerr_f = [[p_f - p_f_lo], [p_f_hi - p_f]]
            ax.errorbar(rec_f, p_f, xerr=xerr_f, yerr=yerr_f, **err_style)

        # 3. Connect Raw -> Filtered
        ax.plot([rec_r, rec_f], [p_r, p_f], color=c, ls='--', lw=4, alpha=0.6, zorder=1)
        
        # 4. Scatter points
        ax.scatter(rec_r, p_r, edgecolors=c, marker=m, facecolors='none', s=350, lw=4, zorder=3)
        ax.scatter(rec_f, p_f, color=c, marker=m, s=350, zorder=3)

    ax.set_xlabel('Recall', fontweight='bold', labelpad=15)
    ax.set_ylabel('Precision', fontweight='bold', labelpad=15)
    ax.set_title('Global Performance', fontweight='bold', pad=25)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, ls='--', alpha=0.5)
    
    alg_handles = [Line2D([0], [0], marker=markers[i], color='w', mfc=ALG_COLOR_MAP[n], label=n, ms=18) 
                   for i, n in enumerate(names)]
    state_handles = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='k', markerfacecolor='none', label='Detection', ms=18, markeredgewidth=4),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='Selection', ms=18)
    ]
    
    # Legend
    leg1 = ax.legend(handles=alg_handles, loc='upper left', title="Algorithm", 
                     fontsize=24, bbox_to_anchor=(0.02, 0.98), title_fontsize=28, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=state_handles, loc='upper left', title="State", 
              fontsize=24, bbox_to_anchor=(0.02, 0.68), title_fontsize=28, framealpha=0.9)

def plot_metrics_table(dfs_class, dfs_global, names, path):
    def fmt(row, metric_key, sub_key, mode='standard'):
        try:
            if metric_key not in row or sub_key not in row[metric_key]: return "-"
            m = row[metric_key]['mean']
            if RealData:
                if mode == 'count': return "{:.0f}".format(m)
                elif mode == 'percent': return "{:.1f}%".format(m*100)
                else: return "{:.3f}".format(m)
            lo, hi = row[metric_key]['p025'], row[metric_key]['p975']
            if mode == 'count': return "{:.0f} [{:.0f}, {:.0f}]".format(m, lo, hi)
            elif mode == 'percent': return "{:.1f}% [{:.1f}, {:.1f}]".format(m*100, lo*100, hi*100)
            else: return "{:.3f} [{:.3f}, {:.3f}]".format(m, lo, hi)
        except: return "-"

    data_left, data_right, alg_row_counts = [], [], []
    for df_c, df_g, name in zip(dfs_class, dfs_global, names):
        num_classes = len(df_c); alg_row_counts.append(num_classes)
        for idx in range(num_classes):
            row = df_c.iloc[idx]
            data_left.append(["", df_c.iloc[idx, 0], fmt(row, 'Raw_Count', 'mean', 'count'), fmt(row, 'Filtered_Count', 'mean', 'count'), fmt(row, 'Retention', 'mean', 'percent')])
        def get_g(m, s):
            r = df_g[df_g.iloc[:, 0] == m]
            return fmt(r.iloc[0], s, 'mean') if not r.empty else "-"
        data_right.append(["Detection", get_g('Global_Precision', 'Raw'), get_g('Global_Recall', 'Raw'), get_g('Global_F1_Score', 'Raw')])
        data_right.append(["Selection", get_g('Global_Precision', 'Filtered'), get_g('Global_Recall', 'Filtered'), get_g('Global_F1_Score', 'Filtered')])

    total_left_rows = sum(alg_row_counts); base_row_height = 1.0 / (total_left_rows + 1)
    fig = plt.figure(figsize=(26, (total_left_rows + 1) * 0.6 + 2.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 0.8], wspace=0.02)
    ax_l, ax_r = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax_l.axis('off'); ax_r.axis('off')

    cols_l = ['Algorithm', 'Class', 'Raw Count', 'Sel. Count', 'Retention']
    tbl_l = ax_l.table(cellText=data_left, colLabels=cols_l, loc='upper center', cellLoc='center', colWidths=[0.15, 0.25, 0.2, 0.2, 0.2])
    cols_r = ['State', 'Precision', 'Recall', 'F1 Score']
    tbl_r = ax_r.table(cellText=data_right, colLabels=cols_r, loc='upper center', cellLoc='center', colWidths=[0.2, 0.26, 0.26, 0.26])

    for tbl in [tbl_l, tbl_r]:
        tbl.auto_set_font_size(False); tbl.set_fontsize(16)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_linewidth(0)
            if r == 0: cell.set_text_props(weight='bold', fontsize=18); cell.set_height(base_row_height)
    
    for r in range(1, len(data_left) + 1):
        for c in range(len(cols_l)): tbl_l[r, c].set_height(base_row_height)
    current_r_idx = 1
    for n_rows in alg_row_counts:
        rh = (n_rows * base_row_height) / 2
        for c in range(len(cols_r)):
            tbl_r[current_r_idx, c].set_height(rh); tbl_r[current_r_idx + 1, c].set_height(rh)
        current_r_idx += 2
        
    accum_rows = 0
    for i, (n_rows, name) in enumerate(zip(alg_row_counts, names)):
        y_center = 1.0 - base_row_height - (accum_rows * base_row_height) - (n_rows * base_row_height) / 2
        ax_l.text(0.07, y_center, name, ha='center', va='center', weight='bold', transform=ax_l.transAxes, fontsize=18)
        accum_rows += n_rows
        if i < len(names) - 1:
            y_line = 1.0 - base_row_height - (accum_rows * base_row_height)
            ax_l.axhline(y_line, c='gray', lw=0.5); ax_r.axhline(y_line, c='gray', lw=0.5)
            
    ax_l.set_title("Class Specific", weight='bold', pad=20, fontsize=24); ax_r.set_title("Global", weight='bold', pad=20, fontsize=24)
    plt.suptitle("Table 1. Comprehensive Benchmarking Statistics.", x=0.1, y=0.98, ha='left', weight='bold', fontsize=28)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()

# --- 4. EXECUTION ---
dfs_class = [prepare_data(pd.read_csv(f, header=[0,1])) for f in class_files]
dfs_global = [pd.read_csv(f, header=[0,1]) for f in global_files]
unique_classes_in_data, data_by_class = organize_data_by_class(dfs_class, alg_names)

sim_types = snakemake.params.sim_types 

# --- INDIVIDUAL PLOTS ---
for i, sim_key in enumerate(sim_types):
    if sim_key not in sim_config: continue
    display_name = sim_config[sim_key].get("display_name", sim_key)
    if display_name in data_by_class:
        if i < len(out_counts):
            fig, ax = plt.subplots(figsize=(12, 10))
            plot_overlapping_counts(display_name, data_by_class[display_name], alg_names, ax, show_ylabel=True)
            bar_legend_handles = [
                Patch(facecolor='gray', alpha=0.25, label='Detection (Total)'),
                Patch(facecolor='gray', alpha=0.6, edgecolor='black', label='Selection (Retained)')
            ]
            ax.legend(handles=bar_legend_handles, loc='best', fontsize=24)
            plt.savefig(out_counts[i], dpi=300, bbox_inches='tight'); plt.close()
        if i < len(out_retention):
            plt.figure(); plt.savefig(out_retention[i]); plt.close()
    else:
        if i < len(out_counts): plt.figure(); plt.savefig(out_counts[i]); plt.close()
        if i < len(out_retention): plt.figure(); plt.savefig(out_retention[i]); plt.close()

# --- PR PLOT (Standalone) ---
fig, ax = plt.subplots(figsize=(15, 15))
plot_global_pr(dfs_global, alg_names, ax)
plt.savefig(out_pr, dpi=300, bbox_inches='tight'); plt.close()

# --- MASTER PANEL (BARS ONLY) ---
n_classes = len(unique_classes_in_data)
h = 10
total_cols = n_classes 
ws = [1] * n_classes 

fig_width = (8 * n_classes) 
fig = plt.figure(figsize=(fig_width, h))

gs = gridspec.GridSpec(2, total_cols, width_ratios=ws, wspace=0.3, hspace=0.3, height_ratios=[1, 0.3]) 

for i, c_name in enumerate(unique_classes_in_data):
    is_first_col = (i == 0)
    
    ax_t = fig.add_subplot(gs[0, i])
    plot_overlapping_counts(c_name, data_by_class[c_name], alg_names, ax_t, 
                            show_ylabel=is_first_col, suppress_title=True)
    
    ax_t.set_title(c_name, weight='bold', fontsize=48, pad=90)
    ax_t.text(-0.25, 1.05, f"{chr(65+i)}", transform=ax_t.transAxes, size=40, weight='bold')

    if is_first_col:
        ax_t.yaxis.set_label_coords(-0.3, 0.5)

# --- LEGEND ---
ax_legend = fig.add_subplot(gs[1, :]) 
ax_legend.axis('off')

bar_legend_handles = [
    Patch(facecolor='gray', alpha=0.25, label='Detection (Total)'),
    Patch(facecolor='gray', alpha=0.6, edgecolor='black', label='Selection (Retained)')
]

ax_legend.legend(handles=bar_legend_handles, loc='center', 
                 fontsize=32, frameon=False, ncol=2)

plt.savefig(out_total, dpi=300, bbox_inches='tight'); plt.close()
plot_metrics_table(dfs_class, dfs_global, alg_names, out_table)
