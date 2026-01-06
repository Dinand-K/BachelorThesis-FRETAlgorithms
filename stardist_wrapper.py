import sys
import os
import numpy as np
import pandas as pd
import yaml
from tifffile import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.measure import regionprops
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Suppress TensorFlow logs for cleaner pipeline output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

#==========
#PARAMETERS
#==========
INPUT_PATH = sys.argv[1]
OUTPUT_DET_PATH = sys.argv[2]
OUTPUT_SEL_PATH = sys.argv[3]

config_path = sys.argv[4]

with open(config_path, 'r') as f:
    full_config = yaml.safe_load(f)

p = full_config["algorithms"]["stardist"]["params"]
p_filt = p["filter"]

# Image Pre-processing
CHANNEL_INDEX = p["channel_index"]
NORM_PERCENTILE_MIN = p["norm_percentile_min"]
NORM_PERCENTILE_MAX = p["norm_percentile_max"]
AXIS_NORM = tuple(p["axis_norm"]) # Convert list from yaml to tuple
# StarDist Model Parameters
STARDIST_MODEL_NAME = p["model_name"]
PROB_THRESHOLD = p["prob_threshold"]
NMS_THRESHOLD = p["nms_threshold"]
MERGING_DIST = p["merging_dist"]
# Filter Parameters
MIN_AREA = p_filt["min_area"]
MAX_AREA = p_filt["max_area"]
MAX_MEAN_INTENSITY = p_filt["max_mean_intensity"]
MAX_ECCENTRICITY = p_filt["max_eccentricity"]
MIN_SOLIDITY = p_filt["min_solidity"]


#==========
#HELPER FUNCTIONS
#==========

def merge_close_points(points, dist_threshold):
    """Merges points within dist_threshold of each other."""
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
#MAIN PIPELINE
#==========

# 1. CORE PROCESSING & PREDICTION
# Load Video and Create Mean Projection
tiffvideo = imread(INPUT_PATH)
tiffvideo = tiffvideo[:, CHANNEL_INDEX, :, :]
mean_video = np.mean(tiffvideo, axis=0)

# Normalize
img = normalize(mean_video, NORM_PERCENTILE_MIN, NORM_PERCENTILE_MAX, axis=AXIS_NORM)

# Load Model (Downloads if not present, otherwise loads from cache)
model = StarDist2D.from_pretrained(STARDIST_MODEL_NAME)

# Predict
print("Running StarDist Prediction...")
labels, details = model.predict_instances(
    img,
    prob_thresh=PROB_THRESHOLD,
    nms_thresh=NMS_THRESHOLD
)


# 2. OUTPUT 1: RAW DETECTIONS
# Get center points directly from StarDist metadata
raw_points = np.array(details['points'])
final_detections = merge_close_points(raw_points, MERGING_DIST)


# 3. OUTPUT 2: SELECTIONS (FILTERING)
# Calculate properties for filtering (requires intensity image)
regions = regionprops(labels, intensity_image=img)
valid_indices = []

for i, props in enumerate(regions):
    # Condition 1: Size
    cond_size = MIN_AREA <= props.area <= MAX_AREA
    
    # Condition 2: Intensity (Lower than threshold)
    cond_intensity = props.mean_intensity < MAX_MEAN_INTENSITY
    
    # Condition 3: Eccentricity (Lower than threshold -> means more circular)
    cond_eccentricity = props.eccentricity < MAX_ECCENTRICITY
    
    # Condition 4: Solidity (Higher than threshold -> means more convex/solid)
    cond_solidity = props.solidity > MIN_SOLIDITY

    # Combine all checks
    if cond_size and cond_intensity and cond_eccentricity and cond_solidity:
        valid_indices.append(i)

# Apply filter
if len(valid_indices) > 0 and len(raw_points) > 0:
    filtered_points = raw_points[valid_indices]
    final_selections = merge_close_points(filtered_points, MERGING_DIST)
else:
    final_selections = np.empty((0, 2))


# 4. SAVE
np.save(OUTPUT_DET_PATH, final_detections)
np.save(OUTPUT_SEL_PATH, final_selections)