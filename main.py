#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path

from cellpose import models
# from cellpose.io import imread
# from cellpose.io import logger_setup
# logger_setup();


#%%

model = models.Cellpose(
    gpu=True,
    model_type="cyto"
    )


