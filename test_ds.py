import os
import scipy.io
from config import RAW_PATHS
from loaders import load_openneuro_subject_session, load_sky_subject

# OpenNeuro check
epochs, labels = load_openneuro_subject_session(RAW_PATHS["brandmeyer"], "sub-001", 1)
print("brandmeyer:", epochs.shape if epochs is not None else "None", labels.shape if labels is not None else "None")

# SKY check
mat = scipy.io.loadmat(os.path.join(RAW_PATHS["sky"], "subjects_epochs.mat"))
timings = mat["timings"]
epochs, labels = load_sky_subject(RAW_PATHS["sky"], 0, timings)
print("SKY:", epochs.shape if epochs is not None else "None", labels.shape if labels is not None else "None") 

from loaders import load_ds3969_subject
epochs, labels = load_ds3969_subject(RAW_PATHS["braboszcz"], "sub-005")
print("barboszcz:", epochs.shape if epochs is not None else "None", labels.shape if labels is not None else "None")