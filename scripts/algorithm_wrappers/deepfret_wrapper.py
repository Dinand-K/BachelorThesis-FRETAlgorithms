import sys
import os
import numpy as np
import tifffile as tiff
import yaml
from tensorflow.keras.models import load_model

# DeepFret path
sys.path.insert(0, "algorithms/DEEPFret/main/python") 

# DeepFRET source code Imports
import lib.imgdata as imgdata
import lib.math as libmath
from lib.imgdata import find_spots
from lib.container import DataContainer, TraceContainer
from global_variables import GlobalVariables as gvars

print("DeepFRET Wrapper Started")

#==========
#PARAMETERS
#==========

TIFF_PATH       = sys.argv[1]
OUTPUT_DET_PATH = sys.argv[2]
OUTPUT_SEL_PATH = sys.argv[3]

CONFIG_PATH     = sys.argv[4]
with open(CONFIG_PATH, 'r') as f:
    full_config = yaml.safe_load(f)

p = full_config["algorithms"]["deepfret"]["params"]
VIDEO_NAME = "snakemake_video"
MODEL_PATH = p["model_path"]
SPOT_THRESHOLD = p["spot_threshold"]
CONFIDENCE_THRESHOLD = p["confidence_threshold"]
ALPHA = p["alpha"]
DELTA = p["delta"]



print(f"DEBUG: I am loading lib.imgdata from: {imgdata.__file__}")
print(f"DEBUG: The spot_threshold passed is: {SPOT_THRESHOLD}")


#==========
#HELPER FUNCTIONS, INCLUDING DEEPFRET CORE LOGIC
#==========

def load_and_detect(tiff_path, video_name, threshold):  
    dc = DataContainer()
    dc.load_video_data(
        path=tiff_path,
        name=video_name,
        view_setup=gvars.key_viewsetupInterleaved,
        alex=False,
        donor_is_left=True,
        donor_is_first=True,
        bg_correction=True,
    )
    vid = dc.videos[video_name]

    # Detect spots (using Green mean image)
    image = vid.grn.mean_nobg
    spots = find_spots(image, threshold, method="peak_local_max")

    spots = np.array(spots, dtype=float)
    if spots.size == 0:
        spots = np.empty((0, 2), dtype=float)
    elif spots.ndim == 1:
        spots = spots.reshape(-1, 2)

    #Variables required for later use
    W = vid.width
    H = vid.height
    
    return vid, spots, W, H

def extract_traces(vid, detected_points, video_name):
    traces_dict = {}
    spots_int = np.round(detected_points).astype(int)

    for n, (y, x) in enumerate(spots_int):
        trace = TraceContainer(
            filename=f"{video_name}_pair_{n}.txt",
            name=f"{video_name}_pair_{n}",
            video=video_name,
            n=n,
        )

        yx = (int(y), int(x))
        try:
            masks = imgdata.circle_mask(yx=yx, indices=vid.indices, **gvars.cmask_p)
            
            # Extract Raw Intensities
            trace.grn.int, trace.grn.bg = imgdata.tiff_stack_intensity(vid.grn.raw, *masks, raw=True)
            trace.acc.int, trace.acc.bg = imgdata.tiff_stack_intensity(vid.acc.raw, *masks, raw=True)

            # Metadata
            trace.frames = np.arange(1, len(trace.grn.int) + 1)
            trace.frames_max = int(trace.frames.max()) if trace.frames.size else 0
            
            # Dummy Red channel (DeepFRET legacy requirement for ALEX)
            nanvals = np.zeros(trace.frames_max) * np.nan
            trace.red.int, trace.red.bg = nanvals, nanvals
            trace.load_successful = True
            
            traces_dict[n] = trace
        except Exception:
            continue
    
    return traces_dict

def predict_batch(traces_dict, model_path, alpha, delta):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
        
    model = load_model(model_path)
    traces_list = list(traces_dict.values())
    
    if not traces_list:
        return traces_dict

    # A. Pre-process
    X_batch_list = []
    valid_indices = []

    for i, trace in enumerate(traces_list):
        # Calculate FRET features (D, A) -> 3D feature set
        xi = np.column_stack(libmath.correct_DA(trace.get_intensities(), alpha, delta))
        
        # Check for empty/zero traces (Fix for Divide-by-Zero)
        if np.max(xi) <= 0:
            xi_safe = np.zeros((100, 2))
            X_batch_list.append(xi_safe)
            valid_indices.append(i)
            continue

        # Crop/Pad to 100 frames
        if xi.shape[0] > 100:
            xi_cropped = xi[:100, [1, 2]] 
        else:
            padding = np.zeros((100 - xi.shape[0], 2))
            xi_cropped = np.vstack([xi[:, [1, 2]], padding])
        #DeepFRET expects shape (Time, 2)
        #xi_cropped = xi[:, [1, 2]]

        # Normalize
        xi_cropped = libmath.sample_max_normalize_3d(X=xi_cropped)

        X_batch_list.append(xi_cropped)
        valid_indices.append(i)

    # B. Stack
    X_batch_np = np.array(X_batch_list)
    if X_batch_np.ndim == 4 and X_batch_np.shape[1] == 1:
            X_batch_np = np.squeeze(X_batch_np, axis=1)

    # C. Predict
    all_predictions = model.predict(X_batch_np, batch_size=64, verbose=0)

    # D. Assign back & Score
    for idx, list_idx in enumerate(valid_indices):
        trace = traces_list[list_idx]
        
        # yi is shape (100, 6)
        yi = all_predictions[idx]
        
        # Sum of Static(4) + Dynamic(5) for each frame
        p_valid_per_frame = np.sum(yi[:, 4:], axis=1) 
        trace.confidence = np.mean(p_valid_per_frame)
        
        # Save raw prediction
        trace.y_pred = yi
        
        # Fill required attributes with dummies so other DeepFRET functions don't crash
        trace.first_bleach = 500 
        for c in trace.channels: c.bleach = 500

    return traces_dict

#==========
#RUNNING DEEPFRET DETECTION AND SELECTION
#==========

# 1. Load
vid_obj, points, W, H = load_and_detect(TIFF_PATH, VIDEO_NAME, SPOT_THRESHOLD)

# 2. Extract
my_traces = extract_traces(vid_obj, points, VIDEO_NAME)

# 3. Predict
my_traces = predict_batch(my_traces, MODEL_PATH, ALPHA, DELTA)

# 4. Filter Results
traces = list(my_traces.values())

if len(traces) > 0:
    # Gather scores
    scores = np.array([t.confidence for t in traces])
    
    # Filter 
    valid_mask = scores >= CONFIDENCE_THRESHOLD
    
    #Offset correction
    x_offset = int(0.02 * W)
    y_offset = int(0.02 * H)
    points[:, 0] += y_offset
    points[:, 1] += x_offset

    pred_valid_points = points[valid_mask, :]

    print(f"DeepFRET Detected: {len(points)}")
    print(f"DeepFRET Selected: {len(pred_valid_points)} (Threshold >= {CONFIDENCE_THRESHOLD})")
    
    # Save
    np.save(OUTPUT_DET_PATH, points)
    np.save(OUTPUT_SEL_PATH, pred_valid_points)
    print(f"Saved results to {OUTPUT_DET_PATH} and {OUTPUT_SEL_PATH}")

else:
    print("No traces extracted. Saving empty arrays.")
    np.save(OUTPUT_DET_PATH, np.empty((0, 2)))
    np.save(OUTPUT_SEL_PATH, np.empty((0, 2)))
