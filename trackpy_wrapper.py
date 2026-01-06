import sys
import numpy as np
import pandas as pd
import tifffile as tiff
import yaml
import trackpy as tp
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

#==========
#PARAMETERS
#==========
INPUT_PATH = sys.argv[1]
OUTPUT_DET_PATH = sys.argv[2]
OUTPUT_SEL_PATH = sys.argv[3]

CONFIG_PATH     = sys.argv[4]
with open(CONFIG_PATH, 'r') as f:
    full_config = yaml.safe_load(f)

p = full_config["algorithms"]["trackpy"]["params"]

# Image Loading
CHANNEL_INDEX = p["channel_index"]
FRAME_LIMIT = p["frame_limit"]

# Detection
DETECTION_FEATURE_SIZE = p["detection_feature_size"]
DETECTION_MINMASS = p["detection_minmass"]
INVERT_IMAGE = p["invert_image"]
TP_PROCESSES = p["processes"]

# Linking
TRACKING_MAX_DISPLACEMENT = p["tracking_max_displacement"]
TRACKING_MEMORY = p["tracking_memory"]

# Filtering
TRACKLENGTH_THRESHOLD = p["tracklength_threshold"]
DISPLACEMENT_THRESHOLD = p["displacement_threshold"]

# Merging
MERGE_DIST_DETECTION = p["merge_dist_detection"]
MERGE_DIST_SELECTION = p["merge_dist_selection"]


#==========
#HELPER FUNCTIONS
#==========

def merge_close_points(points, dist_threshold):
    if len(points) == 0: return points
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=dist_threshold)
    if not pairs: return points
    
    pairs_list = list(pairs)
    row_ind, col_ind = zip(*pairs_list)
    N = len(points)
    graph = csr_matrix((np.ones(len(pairs_list)), (row_ind, col_ind)), shape=(N, N))
    n_components, labels = connected_components(graph, directed=False)
    
    df_temp = pd.DataFrame(points, columns=['y', 'x'])
    df_temp['group_id'] = labels
    return df_temp.groupby('group_id')[['y', 'x']].mean().to_numpy()


#==========
#MAIN LOGIC
#==========

# 1. CORE TRACKING
video = tiff.imread(INPUT_PATH)  # Shape is (T, C, H, W)
frames = video[:, CHANNEL_INDEX, :, :]

allframesdetections = tp.batch(
    frames[:FRAME_LIMIT], 
    DETECTION_FEATURE_SIZE, 
    minmass=DETECTION_MINMASS, 
    invert=INVERT_IMAGE, 
    processes=TP_PROCESSES
)

allframestracks = tp.link(
    allframesdetections, 
    TRACKING_MAX_DISPLACEMENT, 
    memory=TRACKING_MEMORY
)

# Clean index immediately after linking to prevent ambiguity
allframestracks = allframestracks.reset_index(drop=True)


# 2. OUTPUT 1: DETECTIONS
# Get start positions of ALL particles
coords_all = (
    allframestracks.loc[allframestracks.groupby('particle')['frame'].idxmin(), ['y', 'x']]
    .to_numpy()
)

# Merge close points
final_detections = merge_close_points(coords_all, MERGE_DIST_DETECTION)


# 3. OUTPUT 2: SELECTIONS
# A. Apply Track Length Filter
long_tracks = tp.filter_stubs(allframestracks, TRACKLENGTH_THRESHOLD)

if len(long_tracks) > 0:
    long_tracks = long_tracks.reset_index(drop=True)
    
    # B. Calculate Displacement
    df_sorted = long_tracks.sort_values(["particle", "frame"])
    start = df_sorted.groupby("particle")[["y", "x"]].first()
    end   = df_sorted.groupby("particle")[["y", "x"]].last()

    disp_df = end - start
    disp_df["r"] = np.sqrt(disp_df["x"]**2 + disp_df["y"]**2)

    # C. Apply Stationarity Filter
    is_stationary_mask = np.asarray(disp_df["r"] < DISPLACEMENT_THRESHOLD)
    raw_selection_coords = start.to_numpy()[is_stationary_mask]

    # D. Merge close points
    final_selections = merge_close_points(raw_selection_coords, dist_threshold=MERGE_DIST_SELECTION)
else:
    final_selections = np.array([])



#==========
#SAVE
#==========

np.save(OUTPUT_DET_PATH, final_detections)
np.save(OUTPUT_SEL_PATH, final_selections)