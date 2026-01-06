import papylio as pp
import xarray as xr
import numpy as np
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import os
import yaml
import sys
import time
from pathlib import Path


#==========
#HELPTER FUNCTIONS
#==========

def find_particles_fromnc(nc_path):
    if not os.path.exists(nc_path):
        print(f"Warning: {nc_path} not found yet.")
        return np.empty((0, 2))
        
    data = xr.open_dataset(nc_path)
    if "coordinates" not in data:
        return np.empty((0, 2))
        
    coords = data["coordinates"]
    x = coords[:, 0, 0].values
    y = coords[:, 0, 1].values
    return np.column_stack((y, x))

def find_particles_from_memory(file_object, selection_name):
    try:
        mask = file_object.get_variable(selection_name).values
        coords = file_object.coordinates[mask]
        x = coords[:, 0, 0].values
        y = coords[:, 0, 1].values
        return np.column_stack((y, x))
    except Exception as e:
        print(f"Selection extraction failed: {e}")
        return np.empty((0, 2))



#==========
#VARIABLES
#==========

input_path = Path(sys.argv[1]).resolve()
output_det_path = Path(sys.argv[2]).resolve()
output_sel_path = Path(sys.argv[3]).resolve()

config_path = sys.argv[4]

with open(config_path, 'r') as f:
    full_config = yaml.safe_load(f)

p = full_config["algorithms"]["papylio"]["params"]

intensity_total_maximum_threshold = p["intensity_total_max_threshold"]
fraction_difference = p["fraction_difference"]
neighbourhood_size_min = p["neighbourhood_size_min"]
neighbourhood_size_max = p["neighbourhood_size_max"]
projection_max_frame = p["projection_max_frame"]
edge_margin = p["edge_margin"]
gaussian_fit_width = p["gaussian_fit_width"]
rolling_window_size = p["rolling_window_size"]


#==========
#FOLDER CLEANING AND PATH SETUP
#==========

# --- Setup paths ---
script_location = Path(__file__).resolve()

# Go up 2 levels: papylio_wrapper.py -> algorithm_wrappers -> scripts -> Ultimate_Pipeline
project_root = script_location.parents[2] 
papylio_dir = project_root / "algorithms" / "papylio"
papylio_dir.mkdir(parents=True, exist_ok=True)
path_DataSimCopy = papylio_dir / "DataSim_copy.tiff"

# Define strings for Papylio/OS compatibility
nc_path = str(path_DataSimCopy.with_suffix(".nc"))
ave_path = str(papylio_dir / f"{path_DataSimCopy.stem}_ave_f0-500_i0.tif")
path_DataSimCopy_str = str(path_DataSimCopy) 
log_path = str(papylio_dir / "DataSim_copy.log")

# Clean old files ---
for p in [path_DataSimCopy_str, nc_path, ave_path, log_path]:
    if os.path.exists(p):
        os.remove(p)

# waiting a bit
print("Waiting for file cleanup in Papylio programme folder...")
time.sleep(6)



#==========
#PAPYLIO DETECTION
#==========

# Copy input Tiff to Papylio folder
print(f"Reading input: {input_path}")
video = imread(input_path)
imwrite(path_DataSimCopy_str, video)
print(f"Saved copy to: {path_DataSimCopy_str}")

# Initialize Papylio Experiment
exp = pp.Experiment(
    main_path=str(papylio_dir), 
    channels=["donor"],  
    import_all=True
)
if not exp.files:
    raise FileNotFoundError("Papylio did not find the Tiff file copy.")

file = exp.files[0] 

# Apply channel settings
file.movie.channel_arrangement = [[[0]]]
file.movie.channels = file.movie.channels[0:1]

# Find Coordinates
file.find_coordinates(
    channels=["donor"],
    method="by_channel",
    projection_image=dict(
        projection_type="average",
        frame_range=(0, projection_max_frame),
        illumination=0
    ),
    peak_finding=dict(
        method="local-maximum-auto",
        fraction_difference=fraction_difference, 
#        method = "local-maximum",
#        minimum_intensity_difference = 0.15,
        filter_neighbourhood_size_min = neighbourhood_size_min,
        filter_neighbourhood_size_max = neighbourhood_size_max
    ),
    coordinate_optimization=dict(
        coordinates_within_margin=dict(margin=edge_margin),
        coordinates_after_gaussian_fit=dict(gaussian_width=gaussian_fit_width)
    )
)

# Get detections
detected_points = find_particles_fromnc(nc_path)



#==========
#PAPYLIO SELECTION
#==========

file.extract_traces()
intensity_total_rolling = file.intensity_total.rolling(frame=rolling_window_size, center=True).mean().dropna('frame')
selection = (intensity_total_rolling < intensity_total_maximum_threshold).all('frame')
file.set_variable(selection, name='selection_intensity_total')
file.apply_selections('selection_intensity_total')

# Get Selected Points
selected_points = find_particles_from_memory(file, 'selection_intensity_total')

print(f"Papylio Detected: {len(detected_points)}")
print(f"Papylio Selected: {len(selected_points)}")



#==========
#SAVING RESULTS
#==========

print(f"Papylio directory before reset: {os.getcwd()}")
os.chdir(project_root)
print(f"Papylio directory reset to: {os.getcwd()}, the project root.")

# Ensure output directories exist
# We use .resolve() paths here, so these will work regardless of os.chdir,
# but resetting the directory makes the logic cleaner.
output_det_path.parent.mkdir(parents=True, exist_ok=True)
output_sel_path.parent.mkdir(parents=True, exist_ok=True)

np.save(output_det_path, detected_points)
np.save(output_sel_path, selected_points)
print(f"Successfully saved to {output_det_path}")