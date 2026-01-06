import pandas as pd
import numpy as np

# 1. Load snakemake variables
# Inputs
input_class_files  = snakemake.input.class_benchmarks
input_global_files = snakemake.input.global_benchmarks
# Outputs
output_class_stats  = snakemake.output.stats_class
output_global_stats = snakemake.output.stats_global

#2. Helper functions (mean lready exists in pandas)
def p05(x): 
    """Calculates the 5th percentile (lower error bound)."""
    return x.quantile(0.05)

def p95(x): 
    """Calculates the 95th percentile (upper error bound)."""
    return x.quantile(0.95)

# The stats we want to calculate.
# This results in MultiIndex columns (e.g., Recall -> [mean, p05, p95])
stats_funcs = ['mean', p05, p95] #Functions defined for use inside "agg", efficient use of built-in functions. mean is built-in, p05 and p95 are custom.


#3 Process class benchmarks
df_class_all = pd.concat(
    (pd.read_csv(f) for f in input_class_files), 
    ignore_index=True
)
numeric_cols = df_class_all.select_dtypes(include=[np.number]).columns
df_class_agg = df_class_all.groupby('Class')[numeric_cols].agg(stats_funcs)

#4 Process global benchmarks
df_global_all = pd.concat(
    (pd.read_csv(f, index_col=0) for f in input_global_files)
)
df_global_agg = df_global_all.groupby(level=0).agg(stats_funcs)
df_global_agg.index.name = 'Metric'

#5 Save outputs
df_class_agg.to_csv(output_class_stats)
df_global_agg.to_csv(output_global_stats)

