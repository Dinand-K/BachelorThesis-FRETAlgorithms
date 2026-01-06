# --- CONFIGURATION PARSING ---
configpath = "config/realdata.yaml"
configfile: configpath

REPEATS = config.get("repeats", 4)
MATCHING_THRESHOLD = config.get("matching_threshold", 2)
ALGORITHMS = [
    alg_name for alg_name, settings in config["algorithms"].items() 
    if settings.get("activate", False)
]

# --- RULES ---
rule all:
    input:
        "results/the_master_panel.png"

rule generate_data:
    output:
        tiff    = "data_simulation/tifffile/rep{rep}.tiff",
        coords  = "data_simulation/particle_coordinates/rep{rep}.npy",
        classes = "data_simulation/particle_classes/rep{rep}.npy"
    params:
        sim_settings = config["simulation"],
        use_real_data = config.get("use_real_data", False)  #switches between data sim and simply reading provided files.
    script:
        "scripts/data_analysis/data_simulation.py"


rule run_algorithm:
    input:
        "data_simulation/tifffile/rep{rep}.tiff"
    output:
        det = "results/{alg}/detections/rep{rep}.npy",
        sel = "results/{alg}/selections/rep{rep}.npy"
    conda:
        lambda wc: f"envs/{wc.alg}.yaml"
    params:
        script_path = lambda wc: f"scripts/algorithm_wrappers/{wc.alg}_wrapper.py",
        config_path = configpath
    #Look up 'threads' in config, default to 1 if missing
    threads: 
        lambda wc: config["algorithms"][wc.alg].get("threads", 1)
    resources:
        # Pull resource cost dynamically from config, default to 0 if missing
        serial_lock = lambda wc: config["algorithms"][wc.alg].get("serial_lock_cost", 1)
    shell:
        #"python {params.script_path} {input} {output.det} {output.sel} {params.arguments}"
        "python {params.script_path} {input} {output.det} {output.sel} {params.config_path}"

rule matching_benchmarking:
    input:
        detected_points   = "results/{alg}/detections/rep{rep}.npy",
        classified_points = "results/{alg}/selections/rep{rep}.npy",
        groundtruth_detections      = "data_simulation/particle_coordinates/rep{rep}.npy",
        groundtruth_classifications = "data_simulation/particle_classes/rep{rep}.npy"
    output:
        class_benchmarks  = "results/{alg}/class_benchmarks/rep{rep}.csv",
        global_benchmarks = "results/{alg}/global_benchmarks/rep{rep}.csv"
    params:
        matching_threshold = MATCHING_THRESHOLD,
        sim_settings = config["simulation"] #ADDED
    script:
        "scripts/data_analysis/matching_and_benchmarking.py"

rule statistics:
    input:
        # lambda wc: uses the wildcard from the output to determine 'alg'
        class_benchmarks  = lambda wc: expand("results/{alg}/class_benchmarks/rep{rep}.csv", 
                                              alg=wc.alg, rep=range(1, REPEATS+1)),
        global_benchmarks = lambda wc: expand("results/{alg}/global_benchmarks/rep{rep}.csv", 
                                              alg=wc.alg, rep=range(1, REPEATS+1))
    output:
        stats_class  = "results/{alg}/statistics_class.csv",
        stats_global = "results/{alg}/statistics_global.csv"
    script:
        "scripts/data_analysis/statistical_analysis.py"

rule plotting:
    input:
        stats_class  = expand("results/{alg}/statistics_class.csv", alg=ALGORITHMS),
        stats_global = expand("results/{alg}/statistics_global.csv", alg=ALGORITHMS)
    output:
        counts_plots        = expand("results/{alg}/detections_diagram.png", alg=ALGORITHMS),
        retention_plots     = expand("results/{alg}/retention_diagram.png", alg=ALGORITHMS),
        pr_plot             = "results/recall_precision_plot.png",
        total_visualization = "results/the_master_panel.png" 
    params:
        alg_ids = ALGORITHMS,
        # Pull display names dynamically from config
        alg_names = [config["algorithms"][a]["display_name"] for a in ALGORITHMS],
        # ADD THIS: Pass simulation config to identify the Target Class
        sim_settings = config["simulation"]
    script:
        "scripts/data_analysis/plotting.py"