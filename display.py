#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

#%% Parameters ----------------------------------------------------------------

# voxSize = (1, 0.413, 0.413) # zyx
voxSize = (1, 0.826, 0.826) # zyx

#%%

data_path = "D:\local_Camenisch\data"
# stack = io.imread(Path(data_path, "stack.tif"))
stack = io.imread(Path(data_path, "stack_lite.tif"))
probs = io.imread(Path(data_path, "probs.tif"))
masks = io.imread(Path(data_path, "masks.tif"))

#%% 

scale = [voxSize[0] / voxSize[1], 1, 1]
viewer = napari.Viewer()
viewer.add_image(stack, scale=scale)
viewer.add_image(probs, scale=scale)
viewer.add_labels(masks, scale=scale)
