#IMPORTS
import numpy as np
import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
import os
import sys

# --- Snakemake IO ---
path_tiff    = snakemake.output.tiff
path_coords  = snakemake.output.coords
path_classes = snakemake.output.classes
sim_config   = snakemake.params.sim_settings
use_real_data_boolean = snakemake.params.use_real_data
rep = snakemake.wildcards.rep

if use_real_data_boolean:
    print("Data simulation script called to use real data")
    
    src_tiff = "realdata/vid.tiff"
    src_coords = "realdata/coord.npy"
    src_classes = "realdata/class.npy"

    video = tiff.imread(src_tiff)
    video = video[:, [1, 0], :, :]
    
    #videozero = np.zeros_like(video)    
    #final_video = np.stack([video_, video_empty], axis=1)
    
    #EXPERIMENTAL
    #tiff.imwrite(path_tiff, video)
    #instead:
    tiff.imwrite(
    path_tiff, 
    video.astype(np.float32), 
    imagej=True, 
    metadata={'axes': 'TCYX'}
    )

    np.save(path_coords, np.load(src_coords))
    np.save(path_classes, np.load(src_classes))

    print(f"Real data processed and saved to {path_tiff}!")

    sys.exit(0)

print(f"Starting data simulation #{rep}...")

# ==========================================
#           SIMULATION FUNCTIONS
# ==========================================

def generate_valid_simulation_tracks(n_stat, n_rw, n_agg, 
                                     n_frames, img_size, margin, min_distance,
                                     rw_step_size, rw_min_life, rw_max_life):
    """
    Generates tracks for ALL particles simultaneously.
    Prevents overlap by checking the 'Ground Truth' location (Mean for RW) 
    against existing neighbors before accepting a particle.
    """
    total_particles = n_stat + n_rw + n_agg
    if total_particles == 0:
        return np.empty((n_frames,0,2)), np.empty((n_frames,0,2)), np.array([]), np.array([]), np.empty((n_frames,0,2))

    valid_tracks = []
    valid_gt_positions = [] 
    types = [] # 0=Stat, 1=RW, 2=Agg

    type_queue = ([0]*n_stat) + ([1]*n_rw) + ([2]*n_agg)
    np.random.shuffle(type_queue)

    low = float(margin)
    high = float(img_size - margin)
    max_consecutive_fails = int(img_size * 10)
    consecutive_fails = 0

    for p_type in type_queue:
        placed = False
        while not placed:
            if consecutive_fails > max_consecutive_fails:
                raise RuntimeError("Could not place all particles, lower amount of particles.")

            # 1. Candidate Start
            start = np.random.uniform(low, high, size=2)
            
            # 2. Generate Track & GT Point
            track = np.full((n_frames, 2), np.nan)
            gt_point = start.copy() # Default for Stat/Agg

            if p_type == 0 or p_type == 2: # Stationary / Aggregate
                track[:] = start
            
            elif p_type == 1: # Random Walk
                t_start = np.random.randint(0, n_frames)
                life = np.random.randint(rw_min_life, min(rw_max_life, n_frames) + 1)
                t_end = min(t_start + life - 1, n_frames - 1)
                L = t_end - t_start + 1
                
                track[t_start] = start
                if L > 1:
                    steps = np.random.uniform(-rw_step_size, rw_step_size, size=(L - 1, 2))
                    traj = np.cumsum(steps, axis=0) + start
                    traj = np.clip(traj, 0, img_size - 1)
                    track[t_start+1 : t_start+L] = traj
                
                # Check mean position for safety
                valid_segment = track[t_start : t_start+L]
                gt_point = np.mean(valid_segment, axis=0)

            # 3. Distance Check
            if len(valid_gt_positions) > 0:
                existing_arr = np.array(valid_gt_positions)
                dists = np.linalg.norm(existing_arr - gt_point, axis=1)
                
                if np.any(dists < min_distance):
                    consecutive_fails += 1
                    continue 

            # 4. Success
            valid_tracks.append(track)
            valid_gt_positions.append(gt_point)
            types.append(p_type)
            consecutive_fails = 0
            placed = True

    # Reshape to (Time, Particles, 2)
    all_tracks = np.array(valid_tracks) 
    all_tracks = np.moveaxis(all_tracks, 0, 1) 
    types = np.array(types)

    # Slice back into groups
    stat_tracks = all_tracks[:, types==0, :]
    rw_tracks   = all_tracks[:, types==1, :]
    agg_tracks  = all_tracks[:, types==2, :]

    # Recover RW Start/Ends for intensity generation
    rw_starts, rw_ends = [], []
    for i in range(all_tracks.shape[1]):
        if types[i] == 1:
            valid_idx = np.where(~np.isnan(all_tracks[:, i, 0]))[0]
            if len(valid_idx) > 0:
                rw_starts.append(valid_idx[0])
                rw_ends.append(valid_idx[-1])
            else:
                rw_starts.append(0); rw_ends.append(0)

    return stat_tracks, rw_tracks, np.array(rw_starts), np.array(rw_ends), agg_tracks

def generate_stationary_intensities_hmm(n_frames, num_particles, 
                                        pi=None, trans_matrix=None, 
                                        fret_means=(0.2, 0.8), fret_stds=(0.05, 0.05), 
                                        min_total_brightness=1.0, max_total_brightness=3.0, 
                                        rng_seed=None):
    if num_particles == 0: return np.empty((n_frames, 0, 2), dtype=float)

    rng = np.random.default_rng(rng_seed)
    pi = np.asarray(pi, dtype=float) / np.sum(pi)
    trans_matrix = np.asarray(trans_matrix, dtype=float)
    fret_means = np.asarray(fret_means, dtype=float)
    fret_stds = np.asarray(fret_stds, dtype=float)
    
    total_brightness_list = rng.uniform(min_total_brightness, max_total_brightness, size=num_particles)
    intensities = np.empty((n_frames, num_particles, 2), dtype=float)
    cum_trans = np.cumsum(trans_matrix, axis=1)

    for i in range(num_particles):
        # State generation
        r0 = rng.random()
        state0 = 0 if r0 < pi[0] else 1
        states = np.empty(n_frames, dtype=np.int8)
        states[0] = state0
        for t in range(1, n_frames):
            r = rng.random()
            prev = states[t-1]
            states[t] = 0 if r < cum_trans[prev, 0] else 1
            
        # FRET Efficiency with noise
        current_means = fret_means[states]
        current_stds = fret_stds[states]
        efficiency_trace = np.clip(rng.normal(loc=current_means, scale=current_stds, size=n_frames), 0.0, 1.0)
        
        # Intensity Calculation
        total_b = total_brightness_list[i]
        intensities[:, i, 0] = total_b * (1.0 - efficiency_trace) # Donor
        intensities[:, i, 1] = total_b * efficiency_trace         # Acceptor

    return intensities

def generate_moving_intensities(n_frames, num_particles, t_starts, t_ends, min_amp=0.5, max_amp=2.0, fade_len=1):
    if num_particles == 0: return np.empty((n_frames, 0, 2), dtype=float)
        
    intensities = np.full((n_frames, num_particles, 2), np.nan, dtype=float)
    amplitudes = np.random.uniform(min_amp, max_amp, num_particles)
    
    for i in range(num_particles):
        s, e = int(t_starts[i]), int(t_ends[i])
        L = e - s + 1
        fade = min(fade_len, L // 2)
        
        #Fade in and fade out
        rise = np.linspace(0, 1, fade)
        flat = np.ones(max(0, L - 2 * fade))
        fall = np.linspace(1, 0, fade)
        curve = np.concatenate([rise, flat, fall])
        
        trace = amplitudes[i] * curve
        intensities[s : e + 1, i, :] = np.stack([trace, trace], axis=1)
        
    return intensities

def generate_aggregate_intensities(n_frames, num_aggregates, num_fluor_range, 
                                   fluor_brightness_mean, fluor_brightness_var, fluor_pi,
                                   fluor_E_means, fluor_E_stds, fluor_trans_matrix,
                                   jitter_std, rng_seed=None):
    
    rng = np.random.default_rng(rng_seed)
    num_fluor = rng.integers(num_fluor_range[0], num_fluor_range[1] + 1, size=num_aggregates)
    
    # Normalize transition matrix just in case
    fluor_trans_matrix = np.array(fluor_trans_matrix, dtype=float)
    fluor_trans_matrix /= fluor_trans_matrix.sum(axis=1, keepdims=True)

    agg_intensities = np.zeros((n_frames, num_aggregates, 2), dtype=float)
    
    for i in range(num_aggregates):
        N = int(num_fluor[i])
        agg_trace_accum = np.zeros((n_frames, 2), dtype=float)
        
        for j in range(N):
            fluor_seed = None if rng_seed is None else int(rng.integers(0, 2**31 - 1))
            fluor_brightness = max(0.1, rng.normal(loc=fluor_brightness_mean, scale=fluor_brightness_var * fluor_brightness_mean))
            
            single_trace = generate_stationary_intensities_hmm(
                n_frames=n_frames, num_particles=1, pi=fluor_pi, trans_matrix=fluor_trans_matrix,
                fret_means=fluor_E_means, fret_stds=fluor_E_stds,
                min_total_brightness=fluor_brightness, max_total_brightness=fluor_brightness,
                rng_seed=fluor_seed
            )
            agg_trace_accum += single_trace[:, 0, :]
    
        # Apply global jitter
        jitter = rng.normal(loc=1, scale=jitter_std, size=(n_frames,2))
        agg_intensities[:, i, :] = np.clip(agg_trace_accum * jitter, 0.0, None)
        
    return agg_intensities

def render_frames(stationary_positions, rw_positions, agg_positions, 
                  stationary_intensities, rw_intensities, agg_intensities, 
                  num_stationary, num_randomwalk, num_aggregates, 
                  n_frames, img_size, gauss_kernel, gauss_sigma, 
                  photon_scale=1000, read_noise_sigma=3.0, background_level=0.01, 
                  rng_seed=None):
    
    rng = np.random.default_rng(rng_seed)
    frames = []

    # Pre-calculate Gaussian kernel
    radius = int(np.ceil(4 * gauss_sigma))
    y_range = np.arange(-radius, radius + 1)
    x_range = np.arange(-radius, radius + 1)
    dy_grid, dx_grid = np.meshgrid(y_range, x_range, indexing='ij')
    
    # Normalization Factor (Ensures Total Volume = 1.0)
    gaussian_norm_factor = 1.0 / (2 * np.pi * gauss_sigma**2)

    def add_particles(frame, positions, intensities):
        valid_mask = ~np.isnan(positions).any(axis=1)
        valid_pos = positions[valid_mask]
        valid_int = intensities[valid_mask]

        for (x, y), (i_donor, i_acceptor) in zip(valid_pos, valid_int):
            cx, cy = int(np.round(x)), int(np.round(y))
            shift_x, shift_y = cx - x, cy - y
            
            # Sub-pixel Weights
            dist_sq = (dx_grid + shift_x)**2 + (dy_grid + shift_y)**2
            weights = np.exp(-dist_sq / (2 * gauss_sigma**2))
            weights *= gaussian_norm_factor
            
            # Bounds Checking
            y_start, y_end = cy - radius, cy + radius + 1
            x_start, x_end = cx - radius, cx + radius + 1
            w_y_s, w_y_e = 0, weights.shape[0]
            w_x_s, w_x_e = 0, weights.shape[1]
            
            if y_start < 0: w_y_s, y_start = -y_start, 0
            if y_end > img_size: w_y_e, y_end = w_y_e - (y_end - img_size), img_size
            if x_start < 0: w_x_s, x_start = -x_start, 0
            if x_end > img_size: w_x_e, x_end = w_x_e - (x_end - img_size), img_size

            if y_start >= y_end or x_start >= x_end: continue

            patch = weights[w_y_s:w_y_e, w_x_s:w_x_e]
            
            # Add Light
            frame[y_start:y_end, x_start:x_end, 0] += i_donor * patch
            frame[y_start:y_end, x_start:x_end, 1] += i_acceptor * patch

    for t in range(n_frames):
        frame = np.zeros((img_size, img_size, 2), dtype=np.float32)
        
        add_particles(frame, stationary_positions[t], stationary_intensities[t])
        add_particles(frame, rw_positions[t], rw_intensities[t])
        add_particles(frame, agg_positions[t], agg_intensities[t])
        
        # Noise Simulation
        frame += background_level
        mean_photons = np.clip(frame * photon_scale, 0, None)
        photon_counts = rng.poisson(mean_photons).astype(np.float32)
        read_noise = rng.normal(0.0, read_noise_sigma, photon_counts.shape)
        
        frames.append(np.clip(photon_counts + read_noise, 0, None))
        
    return np.array(frames)

# ==========================================
#           PARAMETER PARSING
# ==========================================

# Global
global_cfg = sim_config["global_settings"]
n_frames         = global_cfg["n_frames"]
img_size         = global_cfg["img_size"]
background_level = global_cfg["background_level"]
read_noise       = global_cfg["read_noise"]

# Stationary
stat_cfg = sim_config["stationary"]
num_stationary = stat_cfg["count"] if stat_cfg["activate"] else 0
stat_fret_means   = stat_cfg["params"]["fret_efficiency_means"]
stat_fret_stds    = stat_cfg["params"]["fret_efficiency_stds"]
stat_total_bright = stat_cfg["params"]["total_brightness"]
stat_trans_mat    = np.array(stat_cfg["params"]["trans_matrix"])

# Aggregate
agg_cfg = sim_config["aggregate"]
num_aggregate = agg_cfg["count"] if agg_cfg["activate"] else 0
num_fluor_range       = agg_cfg["params"]["num_fluor_range"]
fluor_brightness_mean = agg_cfg["params"]["fluor_brightness_mean"]
fluor_brightness_var  = agg_cfg["params"]["fluor_brightness_var"]
fluor_pi              = agg_cfg["params"]["fluor_pi"]
fluor_E_means         = agg_cfg["params"]["fluor_E_means"]
fluor_E_stds          = agg_cfg["params"]["fluor_E_stds"]
fluor_trans_matrix    = agg_cfg["params"]["fluor_trans_matrix"]
jitter_std            = agg_cfg["params"]["jitter_std"]

# Moving
mov_cfg = sim_config["moving"]
num_randomwalk = mov_cfg["count"] if mov_cfg["activate"] else 0
mov_amp_range  = mov_cfg["params"]["amplitude"]
mov_step       = mov_cfg["params"]["step_size"]
mov_life_min   = mov_cfg["params"]["min_life"]
mov_life_max   = mov_cfg["params"]["max_life"]

# Settings
margin = 3
min_distance_stat = 7
min_distance_agg  = 4
gauss_kernel = (7,7)
gauss_sigma = 1.5
photon_scale = 500

# ==========================================
#           EXECUTION
# ==========================================

# 1. Generate Positions

max_min_distance = max(min_distance_stat, min_distance_agg)

stationary_positions, rw_positions, rw_starts, rw_ends, agg_positions = generate_valid_simulation_tracks(
    n_stat=num_stationary,
    n_rw=num_randomwalk,
    n_agg=num_aggregate,
    n_frames=n_frames,
    img_size=img_size,
    margin=margin,
    min_distance=max_min_distance,
    rw_step_size=mov_step,
    rw_min_life=mov_life_min,
    rw_max_life=mov_life_max
)

# 2. Generate Intensities
stationary_intensities = generate_stationary_intensities_hmm(
    n_frames=n_frames, num_particles=num_stationary, pi=[0.9, 0.1], trans_matrix=stat_trans_mat,
    fret_means=tuple(stat_fret_means), fret_stds=tuple(stat_fret_stds),
    min_total_brightness=stat_total_bright[0], max_total_brightness=stat_total_bright[1], rng_seed=42
)

rw_intensities = generate_moving_intensities(
    n_frames, num_particles=num_randomwalk, t_starts=rw_starts, t_ends=rw_ends,
    min_amp=mov_amp_range[0], max_amp=mov_amp_range[1], fade_len=0
)

agg_intensities = generate_aggregate_intensities(
    n_frames, num_aggregate, num_fluor_range, fluor_brightness_mean, fluor_brightness_var,
    fluor_pi, fluor_E_means, fluor_E_stds, fluor_trans_matrix, jitter_std, None
)

# 3. Render Frames
frames = render_frames(
    stationary_positions, rw_positions, agg_positions,
    stationary_intensities, rw_intensities, agg_intensities,
    num_stationary, num_randomwalk, num_aggregate,
    n_frames, img_size, gauss_kernel, gauss_sigma,
    photon_scale=photon_scale, read_noise_sigma=read_noise,
    background_level=background_level, rng_seed=None
)

# 4. Save Video
# Move Channel dim to index 1: (T, H, W, C) -> (T, C, H, W)
frames_to_save = np.moveaxis(frames, -1, 1)
tiff.imwrite(
    path_tiff, 
    frames_to_save.astype(np.float32), 
    imagej=True, metadata={'axes': 'TCYX'}
)

# 5. Save Ground Truth
# Stationary: Frame 0 position
if num_stationary > 0:
    stationary_starts_yx = stationary_positions[0][:, [1, 0]]
else:
    stationary_starts_yx = np.empty((0, 2))


# EXPERIMENTAL:
# Random Walk: Mean position over valid track
#if num_randomwalk > 0:
#    rw_means_xy = np.nanmean(rw_positions, axis=0) 
#    rw_starts_yx = rw_means_xy[:, [1, 0]]
#else:
#    rw_starts_yx = np.empty((0, 2))

rw_start_coords_xy = rw_positions[rw_starts, np.arange(num_randomwalk), :]
rw_starts_yx = rw_start_coords_xy[:, [1, 0]]
#EXPERIMENTAL


# Aggregate: Frame 0 position
if num_aggregate > 0:
    agg_starts_yx = agg_positions[0][:, [1, 0]]
else:
    agg_starts_yx = np.empty((0, 2))

to_stack = []
if num_stationary > 0: to_stack.append(stationary_starts_yx)
if num_aggregate > 0: to_stack.append(agg_starts_yx)
if num_randomwalk > 0: to_stack.append(rw_starts_yx)

if to_stack:
    full_start_positions = np.vstack(to_stack)
else:
    full_start_positions = np.empty((0, 2))

np.save(path_coords, full_start_positions.astype(np.float32))

# 6. Save Classes
stationary_classes = np.array([0] * num_stationary)
agg_classes = np.array([1] * num_aggregate)
rw_classes = np.array([2] * num_randomwalk)
full_classes = np.concatenate([stationary_classes, agg_classes, rw_classes]) 
np.save(path_classes, full_classes)

print(f"Data simulation #{rep} completed and saved.")


#Printing SNR:
avg_stat_brightness = np.mean(stat_total_bright)
# 1. Get Signal
# Convert total volume to peak height: Height = Volume / (2 * pi * sigma^2)
peak_height_norm = 1.0 / (2 * np.pi * gauss_sigma**2)
S_peak_photons = avg_stat_brightness * peak_height_norm * photon_scale
# 2. Get Background (B)
B_photons = background_level * photon_scale
# 3. Calculate SNR 
# Formula: S / sqrt(S + B + ReadNoise^2)
# Note: (S + B) is the Photon Shot Noise Variance, as shot noise follows poisson statistics, where the mean = variance.
snr = S_peak_photons / np.sqrt(S_peak_photons + B_photons + read_noise**2)
print(f"SNR SIMULATION PROFILE: {snr:.2f}")
