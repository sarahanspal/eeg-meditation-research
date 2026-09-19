DATA_ROOT = "path/to/your/local/eeg-meditation-data"

RAW_PATHS = {
    "brandmeyer": f"{DATA_ROOT}/openneuro ",
    "sky":        f"{DATA_ROOT}/sky+control",
    "mantra":     f"{DATA_ROOT}/mantra-ds",
    "braboszcz":  f"{DATA_ROOT}/delorme2",
}

PROCESSED_ROOT = f"{DATA_ROOT}/eeg-processed"
FEATURES_HANDCRAFTED_ROOT = f"{DATA_ROOT}/features-handcrafted"