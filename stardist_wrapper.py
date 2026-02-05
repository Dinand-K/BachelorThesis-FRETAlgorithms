import sys
import os
import numpy as np
import pandas as pd
import yaml
from tifffile import imread
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from skimage.measure import regionprops
from skimage.morphology import white_tophat, disk
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Suppress TensorFlow logs
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
AXIS_NORM = tuple(p["axis_norm"])

# StarDist Model Parameters
STARDIST_MODEL_NAME = p["model_name"]
PROB_THRESHOLD = p["prob_threshold"]
NMS_THRESHOLD = p["nms_threshold"]
MERGING_DIST = p["merging_dist"]

# Filter Parameters
TOPHAT_RADIUS = p_filt["tophat_radius"]
MIN_AREA = p_filt["min_area"]
MAX_AREA = p_filt["max_area"]
MAX_MEAN_INTENSITY = p_filt["max_mean_intensity"]
MIN_MEAN_INTENSITY = p_filt["min_mean_intensity"]
MAX_ECCENTRICITY = p_filt["max_eccentricity"]
MIN_SOLIDITY = p_filt["min_solidity"]



#==========
#MAIN PIPELINE
#==========

# 1. CORE PROCESSING & PREDICTION
tiffvideo = imread(INPUT_PATH)

# Channel summing
tiffvideo = tiffvideo[:,0,:,:] + tiffvideo[:,1,:,:]
mean_video = np.mean(tiffvideo, axis=0)

# Background correction
mean_video = white_tophat(mean_video, footprint=disk(TOPHAT_RADIUS))

# Normalize
img = normalize(mean_video, NORM_PERCENTILE_MIN, NORM_PERCENTILE_MAX, axis=AXIS_NORM)

# Load Model
model = StarDist2D.from_pretrained(STARDIST_MODEL_NAME)

# Predict
print("Running StarDist Prediction...")
labels, details = model.predict_instances(
    img,
    prob_thresh=PROB_THRESHOLD,
    nms_thresh=NMS_THRESHOLD
)

# 2. OUTPUT 1: RAW DETECTIONS (Using Weighted Centroids)
# Calculate regionprops first to get sub-pixel weighted centroids
regions = regionprops(labels, intensity_image=img)

# Extract weighted centroids (y, x)
if len(regions) > 0:
    raw_points = np.array([props.weighted_centroid for props in regions])
else:
    raw_points = np.empty((0, 2))

final_detections = raw_points

# 3. OUTPUT 2: SELECTIONS (FILTERING)
valid_indices = []

for i, props in enumerate(regions):
    cond_size = MIN_AREA <= props.area <= MAX_AREA
    cond_intensity1 = props.mean_intensity < MAX_MEAN_INTENSITY
    cond_intensity2 = props.mean_intensity > MIN_MEAN_INTENSITY
    cond_eccentricity = props.eccentricity < MAX_ECCENTRICITY
    cond_solidity = props.solidity > MIN_SOLIDITY

    if cond_size and cond_intensity1 and cond_intensity2 and cond_eccentricity and cond_solidity:
        valid_indices.append(i)

if len(valid_indices) > 0:
    filtered_points = raw_points[valid_indices]
    final_selections = filtered_points
else:
    final_selections = np.empty((0, 2))


# 4. SAVE
np.save(OUTPUT_DET_PATH, final_detections)
np.save(OUTPUT_SEL_PATH, final_selections)
