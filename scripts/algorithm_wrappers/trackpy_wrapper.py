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
#frames = video[:, CHANNEL_INDEX, :, :]
#Experimenting with adding both channels --> Works really well
frames = video[:, 0, :, :].astype(np.float32) + video[:, 1, :, :].astype(np.float32)

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
#allframestracks = allframestracks.reset_index(drop=True)

# 2. OUTPUT 1: DETECTIONS
allframestracks = allframestracks.reset_index(drop=True)
coords_all = allframestracks.sort_values('frame').drop_duplicates('particle', keep='first')[['y', 'x']].to_numpy()

final_detections = merge_close_points(coords_all, MERGE_DIST_DETECTION)


# 3. OUTPUT 2: SELECTIONS
# A. Apply Track Length Filter
long_tracks = tp.filter_stubs(allframestracks, TRACKLENGTH_THRESHOLD)

if len(long_tracks) > 0:
    # 1. Clean Index
    long_tracks = long_tracks.reset_index(drop=True)
    # 2. Calculate MSD
    # mpp=1.0, fps=1.0 keeps units in pixels and frames
    imsd = tp.imsd(long_tracks, mpp=1.0, fps=1.0, max_lagtime=20)
    
    # 3. Extract Metric: imsd at lagtime 5
    if 5 in imsd.index:
        metric_per_particle = imsd.loc[5]
    else:
        # Fallback to the largest available lag time if 5 isn't reached
        metric_per_particle = imsd.iloc[-1]
    
    avg_displacement = np.sqrt(metric_per_particle)
    
    # 4. Apply Stationarity Filter
    # Get IDs of particles that moved LESS than threshold
    stationary_ids = avg_displacement[avg_displacement < DISPLACEMENT_THRESHOLD].index
    
    # 5. Extract Start Coordinates
    # Filter long_tracks to only stationary particles, then grab first row
    t_stationary = long_tracks[long_tracks['particle'].isin(stationary_ids)]

    raw_selection_coords = (
        t_stationary
        .sort_values('frame')
        .drop_duplicates('particle', keep='first')[['y', 'x']]
        .to_numpy()
    )

    # 6. Merge close points
    final_selections = merge_close_points(raw_selection_coords, dist_threshold=MERGE_DIST_SELECTION)

else:
    final_selections = np.array([])


#==========
#SAVE
#==========

np.save(OUTPUT_DET_PATH, final_detections)
np.save(OUTPUT_SEL_PATH, final_selections)
