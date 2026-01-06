#IMPORTS
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt

# --- Snakemake IO ---
path_tiff    = snakemake.output.tiff
path_coords  = snakemake.output.coords
path_classes = snakemake.output.classes
sim_config   = snakemake.params.sim_settings
use_real_data_boolean = snakemake.params.use_real_data
rep = snakemake.wildcards.rep

if use_real_data_boolean:
    print("Data simulation script called to use real data")

    import os
    import sys

    # Source paths real data
    src_tiff = "realdata/vid.tiff"
    src_coords = "realdata/coord.npy"
    src_classes = "realdata/class.npy"

    # Safety check
    if not os.path.exists(src_tiff):
        sys.exit(f"Error: Real data mode is active, but real data source file '{src_tiff}' was not found.")

    # 1. TIFF Processing
    # Load raw video. Assuming shape (T, H, W)
    video_raw = tiff.imread(src_tiff)
    
    # Logic to simulate dual-view: Roll the image by half height to create 2nd channel
    # Assuming standard Tiff dims: (Time, Y, X) -> axis 1 is Y (Height)
    if video_raw.ndim >= 2:
        height_axis = 1 if video_raw.ndim == 3 else 0
        height = video_raw.shape[height_axis]
        
        # Create 'Swapped' channel
        video_swapped = np.roll(video_raw, height // 2, axis=height_axis)
        
        # Stack to create (Time, Channel, Y, X)
        final_video = np.stack([video_raw, video_swapped], axis=1)
    else:
        # Fallback for unexpected shapes
        final_video = video_raw

    # Save formatted TIFF
    tiff.imwrite(path_tiff, final_video)

    # 2. Coordinates (Load source -> Save to destination)
    if os.path.exists(src_coords):
        np.save(path_coords, np.load(src_coords))
    else:
        print(f"Warning: real data path {src_coords} not found. Saving empty coordinates.")
        np.save(path_coords, np.empty((0, 2)))

    # 3. Classes (Load source -> Save to destination)
    if os.path.exists(src_classes):
        np.save(path_classes, np.load(src_classes))
    else:
        print(f"Warning: real data path {src_classes} not found. Saving empty classes.")
        np.save(path_classes, np.empty((0,)))

    print(f"Real data processed and saved to {path_tiff}!")
    
    # CRITICAL: Exit script here to prevent the Simulation logic below from running
    sys.exit(0)




print(f"Starting data simulation #{rep}...")

#Functions for data simulation
def generate_stationary_positions(num_stationary, n_frames, img_size, margin, min_distance):
    start_positions_list = []
    max_total_attempts = num_stationary * 1000
    attempts = 0
    while len(start_positions_list) < num_stationary:
        if attempts > max_total_attempts:
            raise RuntimeError(f"Failed to place all particles.")
        candidate_pos = np.random.randint(margin, img_size - margin, size=2)
        attempts += 1
        is_valid = True
        for pos in start_positions_list:
            dist = np.linalg.norm(candidate_pos - pos)
            if dist < min_distance:
                is_valid = False
                break
        if is_valid:
            start_positions_list.append(candidate_pos)
    start_positions = np.array(start_positions_list)
    positions = np.repeat(start_positions[None, :, :], n_frames, axis=0)
    return positions, start_positions

def generate_aggregate_positions(num_aggregate, n_frames, img_size, margin):
    start_positions = np.random.randint(margin, img_size - margin, size=(num_aggregate, 2))
    positions = np.repeat(start_positions[None, :, :], n_frames, axis=0)
    return positions, start_positions

def generate_randomwalk_positions(num_random, n_frames, img_size, step_size, margin, min_life=2, max_life=None):
    if max_life is None: raise ValueError("max_life must be provided.")
    positions = np.full((n_frames, num_random, 2), np.nan, dtype=float)
    t_starts = np.random.randint(0, n_frames, size=num_random)
    lifetimes = np.random.randint(min_life, min(max_life, n_frames) + 1, size=num_random)
    t_ends = np.minimum(t_starts + lifetimes - 1, n_frames - 1)
    start_positions = np.full((num_random, 2), np.nan, dtype=float)
    for i in range(num_random):
        s, e = int(t_starts[i]), int(t_ends[i])
        L = e - s + 1
        start_pos = np.random.randint(margin, img_size - margin, size=2).astype(float)
        start_positions[i] = start_pos
        positions[s, i] = start_pos
        if L > 1:
            steps = np.random.uniform(-step_size, step_size, size=(L - 1, 2))
            traj = np.cumsum(steps, axis=0) + start_pos
            traj = np.clip(traj, 0, img_size - 1)
            positions[s + 1: s + 1 + (L - 1), i] = traj
    return positions, t_starts, t_ends, start_positions



def generate_stationary_intensities_hmm(n_frames, num_particles, 
                                        pi=None, trans_matrix=None, 
                                        fret_means=(0.2, 0.8), fret_stds=(0.05, 0.05), 
                                        min_total=1.0, max_total=3.0, 
                                        rng_seed=None):
    
    rng = np.random.default_rng(rng_seed)
    
    # 1. Setup HMM probabilities
    pi = np.asarray(pi, dtype=float) / np.sum(pi)

    trans_matrix = np.asarray(trans_matrix, dtype=float)
    
    fret_means = np.asarray(fret_means, dtype=float)
    fret_stds = np.asarray(fret_stds, dtype=float)
    
    # 2. Assign Total Brightness (D + A) for each particle
    # This bucket of energy stays roughly constant while FRET fluctuates
    total_brightness_list = rng.uniform(min_total, max_total, size=num_particles)
    
    # Shape: (n_frames, num_particles, 2)
    intensities = np.empty((n_frames, num_particles, 2), dtype=float)
    cum_trans = np.cumsum(trans_matrix, axis=1)

    for i in range(num_particles):
        # --- A) Generate the State Sequence (Markov Chain) ---

        #STarting states
        r0 = rng.random()
        state0 = 0 if r0 < pi[0] else 1
        states = np.empty(n_frames, dtype=np.int8)
        states[0] = state0
        
        #States over time
        for t in range(1, n_frames):
            r = rng.random()
            prev = states[t-1]
            # transition based on previous state
            states[t] = 0 if r < cum_trans[prev, 0] else 1
            
        # --- B) Generate FRET Efficiency (E) with Noise ---
        current_means = fret_means[states]
        current_stds = fret_stds[states]
        
        # This is the "wobble" of the molecule
        efficiency_trace = rng.normal(loc=current_means, scale=current_stds, size=n_frames)
        
        # Clip E to [0, 1] because transfer cannot be >100% or <0%
        efficiency_trace = np.clip(efficiency_trace, 0.0, 1.0)
        
        # --- C) Split the Total Energy based on E ---
        total_b = total_brightness_list[i]
        
        # Donor = Total * (1 - E)
        dd_trace = total_b * (1.0 - efficiency_trace)
        
        # Acceptor = Total * E
        da_trace = total_b * efficiency_trace
        
        # Stack into the last dimension
        intensities[:, i, 0] = dd_trace
        intensities[:, i, 1] = da_trace

    return intensities


def generate_moving_intensities(n_frames, num_particles, t_starts, t_ends, min_amp=0.5, max_amp=2.0, fade_len=5):
    intensities = np.full((n_frames, num_particles, 2), np.nan, dtype=float)
    amplitudes = np.random.uniform(min_amp, max_amp, num_particles)
    
    for i in range(num_particles):
        s, e = int(t_starts[i]), int(t_ends[i])
        A = amplitudes[i]
        L = e - s + 1
        fade = min(fade_len, L // 2)
        rise = np.linspace(0, 1, fade)
        flat = np.ones(max(0, L - 2 * fade))
        fall = np.linspace(1, 0, fade)
        curve = np.concatenate([rise, flat, fall])
        trace = A * curve
        
        trace_duplicated = np.stack([trace, trace], axis=1)
        intensities[s : e + 1, i, :] = trace_duplicated
    return intensities


import numpy as np

def generate_aggregate_intensities(n_frames, num_aggregates, 
                                   fluor_range=(3, 10), 
                                   A0=1.0, 
                                   pi=(0.98, 0.02), 
                                   state_means=(0.2, 0.8), # Renamed defaults to typical FRET Efficiencies 
                                   state_stds=(0.05, 0.05), 
                                   per_agg_amp_jitter=(0.9, 1.1), # not per fluorophore now but for the entire aggregate
                                   rng_seed=None):
    
    rng = np.random.default_rng(rng_seed)
    
    # Randomly decide how many fluorophores are in each aggregate
    N_per_agg = rng.integers(fluor_range[0], fluor_range[1] + 1, size=num_aggregates)
    
    # Hardcoded transition matrix
    trans_matrix = np.array([[0.995, 0.005], [0.01,  0.99]], dtype=float)
    # Ensuring it is normalized
    trans_matrix /= trans_matrix.sum(axis=1, keepdims=True)
    
    # Output shape: (n_frames, num_aggregates, 2)
    agg_intensities = np.zeros((n_frames, num_aggregates, 2), dtype=float)
    
    for i in range(num_aggregates):
        N = int(N_per_agg[i])
        
        # Accumulator for this single aggregate (Donor, Acceptor)
        agg_trace_accum = np.zeros((n_frames, 2), dtype=float)
        
        for j in range(N):
            per_fluo_seed = None if rng_seed is None else int(rng.integers(0, 2**31 - 1))
            
            # Generate a single FRET pair trace
            # We map A0 to the Total Brightness
            # We map state_means to FRET Efficiencies
            single_trace = generate_stationary_intensities_hmm(
                n_frames=n_frames, 
                num_particles=1, 
                pi=pi, 
                trans_matrix=trans_matrix,
                fret_means=state_means,     # Use these as Efficiency states (0.0 - 1.0)
                fret_stds=state_stds, 
                min_total=A0,               # Fixed brightness A0
                max_total=A0, 
                rng_seed=per_fluo_seed
            )
            
            # single_trace shape is (n_frames, 1, 2). 
            # Add it to the accumulator.
            agg_trace_accum += single_trace[:, 0, :]
    
        # Apply the global jitter for this aggregate, instead of per-fluorophore. 
        jitter = rng.uniform(per_agg_amp_jitter[0], per_agg_amp_jitter[1])
        agg_intensities[:, i, :] = agg_trace_accum * jitter
        
    return agg_intensities, N_per_agg

def render_frames(stationary_positions, rw_positions, agg_positions, 
                  stationary_intensities, rw_intensities, agg_intensities, 
                  num_stationary, num_randomwalk, num_aggregates, 
                  n_frames, img_size, gauss_kernel, gauss_sigma, 
                  photon_scale=1000, read_noise_sigma=3.0, background_level=0.01, 
                  rng_seed=None):
    
    rng = np.random.default_rng(rng_seed)
    frames = []
    
    for t in range(n_frames):
        # Initialize a 2-channel canvas (Green/Red)
        # Shape: (Height, Width, Channels)
        frame = np.zeros((img_size, img_size, 2), dtype=np.float32)
        
        # 1. Place Stationary Particles (FRET Pairs)
        for i in range(num_stationary):
            x, y = stationary_positions[t, i]
            I = stationary_intensities[t, i] # Shape (2,) -> [Donor, Acceptor]
            
            x_i, y_i = int(x), int(y)
            if 0 <= x_i < img_size and 0 <= y_i < img_size:
                frame[y_i, x_i, :] += I
                
        # 2. Place Random Walk Particles (Trash)
        for i in range(num_randomwalk):
            # Skip if the particle has left the simulation (NaN)
            if np.isnan(rw_positions[t, i]).any(): continue
            
            x, y = rw_positions[t, i]
            I = rw_intensities[t, i]         # Shape (2,) -> [Trash, Trash]
            
            x_i, y_i = int(x), int(y)
            if 0 <= x_i < img_size and 0 <= y_i < img_size:
                frame[y_i, x_i, :] += I
                
        # 3. Place Aggregates (Clumps of FRET Pairs)
        for i in range(num_aggregates):
            x, y = agg_positions[t, i]
            I = agg_intensities[t, i]        # Shape (2,)
            
            x_i, y_i = int(x), int(y)
            if 0 <= x_i < img_size and 0 <= y_i < img_size:
                frame[y_i, x_i, :] += I
                
        # --- Simulate Optics & Camera Noise ---
        
        # 1. Diffraction Limit (PSF)
        # cv2.GaussianBlur applies the blur independently to each channel
        frame = cv2.GaussianBlur(frame, gauss_kernel, gauss_sigma)
        
        # 2. Add Background (Uniform fluorescence in solution)
        # We assume background is roughly equal or just added to both
        frame = frame + background_level
        frame = np.clip(frame, 0.0, None)
        
        # 3. Photon Shot Noise (Poisson Statistics)
        # Convert arbitrary units -> Mean Photon Count
        mean_photons = frame * photon_scale
        
        # Sample actual photons detected (Independent for Ch1 and Ch2)
        photon_counts = rng.poisson(mean_photons).astype(np.float32)
        
        # 4. Camera Read Noise (Gaussian Electronic Noise)
        # Independent noise for every pixel in every channel
        read_noise = rng.normal(0.0, read_noise_sigma, photon_counts.shape)
        photon_counts += read_noise
        
        # Clip to 0 (cannot have negative photons)
        photon_counts = np.clip(photon_counts, 0, None)
        
        # Convert back to arbitrary units (optional, often kept as counts in raw data)
        frame_final = photon_counts / float(photon_scale)
        
        frames.append(frame_final)
        
    return np.array(frames) # Returns shape (T, H, W, 2)

# ==========================================
#       PARAMETER PARSING FROM CONFIG
# ==========================================

# 1. Global
global_cfg = sim_config["global_settings"]
n_frames         = global_cfg["n_frames"]
img_size         = global_cfg["img_size"]
background_level = global_cfg["background_level"]
read_noise       = global_cfg["read_noise"]

# 2. Stationary
stat_cfg = sim_config["stationary"]
num_stationary = stat_cfg["count"] if stat_cfg["activate"] else 0
# Renamed variables
stat_fret_means  = stat_cfg["params"]["fret_efficiency_means"]
stat_fret_stds   = stat_cfg["params"]["fret_efficiency_stds"]
stat_total_bright= stat_cfg["params"]["total_brightness"]
stat_trans_mat   = np.array(stat_cfg["params"]["trans_matrix"])

# 3. Aggregate
agg_cfg = sim_config["aggregate"]
num_aggregate = agg_cfg["count"] if agg_cfg["activate"] else 0
# Renamed variables
agg_fret_means   = agg_cfg["params"]["fret_efficiency_means"]
agg_fluor_rng    = agg_cfg["params"]["fluor_range"]
agg_total_bright = agg_cfg["params"]["total_brightness"][0]

# 4. Moving
mov_cfg = sim_config["moving"]
num_randomwalk = mov_cfg["count"] if mov_cfg["activate"] else 0
mov_amp_range  = mov_cfg["params"]["amplitude"]
mov_step       = mov_cfg["params"]["step_size"]
mov_life_min   = mov_cfg["params"]["min_life"]
mov_life_max   = mov_cfg["params"]["max_life"]

# 5. Hardcoded settings
margin = 3
min_distance_stat = 7
min_distance_agg  = 4
gauss_kernel = (7,7)
gauss_sigma = 1.5
photon_scale = 80


# ==========================================
#           EXECUTION
# ==========================================

# Stationary Positions
stationary_positions, stationary_start_positions = generate_stationary_positions(
    num_stationary = num_stationary,
    n_frames = n_frames,
    img_size = img_size,
    margin = margin,
    min_distance = min_distance_stat
)

# Random Walk positions
rw_positions, rw_starts, rw_ends, rw_start_positions = generate_randomwalk_positions(
    num_random = num_randomwalk,
    n_frames = n_frames,
    img_size = img_size,
    step_size = mov_step,
    margin = margin,
    min_life = mov_life_min,
    max_life = mov_life_max,
)

# Aggregate Positions
agg_positions, aggregate_start_positions = generate_stationary_positions(
    num_stationary = num_aggregate,
    n_frames = n_frames,
    img_size = img_size,
    margin = margin,
    min_distance = min_distance_agg
)

# Stationary Intensities 
stationary_intensities = generate_stationary_intensities_hmm(
    n_frames=n_frames, 
    num_particles=num_stationary,
    pi=[0.9, 0.1], 
    trans_matrix=stat_trans_mat,
    fret_means=tuple(stat_fret_means),
    fret_stds=tuple(stat_fret_stds),
    min_total=stat_total_bright[0],
    max_total=stat_total_bright[1],
    rng_seed=42
)

# Random walk intensities (Trash/Contaminants)
rw_intensities = generate_moving_intensities(
    n_frames,
    num_particles=num_randomwalk,
    t_starts=rw_starts,
    t_ends=rw_ends,
    min_amp=mov_amp_range[0],
    max_amp=mov_amp_range[1],
    fade_len=5,
)

# Aggregate intensities
agg_intensities, N_per_agg = generate_aggregate_intensities(
    n_frames=n_frames,
    num_aggregates=num_aggregate,
    fluor_range=tuple(agg_fluor_rng),
    A0=agg_total_bright,
    pi=(0.98, 0.02),
    state_means=tuple(agg_fret_means),
    state_stds=(0.04, 0.04), # Hardcoded wobble for aggregates
    per_agg_amp_jitter=(0.8, 1.5),
    rng_seed=42
)

# --- Render frames ---
frames = render_frames(
    stationary_positions, 
    rw_positions,
    agg_positions,
    stationary_intensities,
    rw_intensities,
    agg_intensities,
    num_stationary,
    num_randomwalk,
    num_aggregate,
    n_frames,
    img_size,
    gauss_kernel=gauss_kernel,
    gauss_sigma=gauss_sigma,
    photon_scale=photon_scale, 
    read_noise_sigma=read_noise,
    background_level=background_level,
    rng_seed=None,
)

# Current shape from render function: (n_frames, img_size, img_size, 2) -> (T, H, W, C)
# Move the Channel dimension to index 1, so new shape: (n_frames, 2, img_size, img_size) -> (T, C, H, W), which is the standard.
frames_to_save = np.moveaxis(frames, -1, 1)

# Save video
tiff.imwrite(
    path_tiff, 
    frames_to_save.astype(np.float32),  #32-bit float
    imagej=True,                        #Tells ImageJ to handle as hyperstack
    metadata={'axes': 'TCYX'}           #Defines axes for ImageJ
)

# Prepare and save Ground Truth positions
stationary_starts_yx = stationary_start_positions[:, [1, 0]].astype(int) 
rw_starts_yx = rw_start_positions[:, [1, 0]].astype(int)
agg_starts_yx = aggregate_start_positions[:, [1, 0]].astype(int)

# Stack must handle empty arrays if counts are 0
to_stack = []
if num_stationary > 0: to_stack.append(stationary_starts_yx)
if num_aggregate > 0: to_stack.append(agg_starts_yx)
if num_randomwalk > 0: to_stack.append(rw_starts_yx)

if to_stack:
    full_start_positions = np.vstack(to_stack)
else:
    full_start_positions = np.empty((0, 2))

np.save(path_coords, full_start_positions)

# Prepare and save Ground Truth classifications
stationary_classes = np.array([0] * num_stationary)  # stationary = 0
agg_classes = np.array([1] * num_aggregate)          # aggregate = 1
rw_classes = np.array([2] * num_randomwalk)          # random-walk = 2
full_classes = np.concatenate([stationary_classes, agg_classes, rw_classes]) 
np.save(path_classes, full_classes)

print(f"Data simulation #{rep} completed and saved.")