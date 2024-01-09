#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
from cellpose import models

#%% Parameters ----------------------------------------------------------------

# voxSize = (1, 0.413, 0.413) # zyx
voxSize = (1, 0.826, 0.826) # zyx

#%%

data_path = "D:\local_Camenisch\data"
# stack = io.imread(Path(data_path, "stack.tif"))
stack = io.imread(Path(data_path, "stack_lite.tif"))

#%% 

img = stack[70:130,...]

#%% 

print("  Predict :", end='')
t0 = time.time()

model = models.Cellpose(model_type="cyto", gpu=True)
# model = models.CellposeModel(model_type='tissuenet', gpu=True)

masks, flows, styles, diams = model.eval(
    img, 
    batch_size=4,
    diameter=10, 
    channels=[0,0],
    flow_threshold=0.8, 
    cellprob_threshold=-3,
    # stitch_threshold=0.05,
    
    do_3D=True,
    anisotropy=voxSize[0] / voxSize[1],
    min_size=128,
    
    )

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")

#%%

scale = [voxSize[0] / voxSize[1], 1, 1]
viewer = napari.Viewer()
viewer.add_image(img, scale=scale)
viewer.add_labels(masks, scale=scale)
viewer.add_image(flows[2], scale=scale)

# viewer.add_image(stack, scale=scale)