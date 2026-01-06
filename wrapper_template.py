# 1. IMPORTS
# Import necessary libraries
import numpy as np
import sys

# Import neccesary libraries from source code
sys.path.insert(0, "algorithms/[your_algorithm_path_here]") #If your model requires imports from a source code, add the path here
from (specified_algorithm_module) import (necessary_classes_or_functions)



# 2. PARAMETERS
#These are always the same:
TIFF_PATH = sys.argv[1] #Path from which to read the data simulation tiff file
output_det_path = sys.argv[2]  #Path to which the detected points are to be saved, as np array in the order [y,x] with shape (num_detections, 2)
#If your model selects points after an initial detection step, you can save those selected points using output_sel_path, to gain insight into the effectiveness of your selection algorithm.
output_sel_path = sys.argv[3]  #Path to which the selectedpoints are to be saved, as np array in the order [y,x] with shape (num_selected, 2)

#These parameters are specified by your algorithm and chosen by you:
#Either pass them from the config file as such:
some_snakemake_param = sys.argv[4]
some_other_snakemake_param = sys.argv[5]
#Or hardcode them in this wrapper:
some_threshold = 0.5
some_variable = 10



# 3. ALGORITHM IMPLEMENTATION
#-----------
#YOUR CODE HERE
#-----------



# 4. SAVE DETECTIONS AND SELECTIONS
np.save(output_det_path, detections)
np.save(output_sel_path, selections)