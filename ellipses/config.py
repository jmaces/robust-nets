import os

from data_management import sample_ellipses


DATA_PATH = os.path.join("raw_data")
RESULTS_PATH = os.path.join("results")

# ----- random seeds -----
torch_seed = 1
numpy_seed = 2
matrix_seed = 3

# ----- signal configuration -----
n = (256, 256)  # signal dimension
data_params = {  # additional data generation parameters
    "c_min": 7,
    "c_max": 15,
    "max_axis": 0.45,
    "min_axis": 0.05,
    "margin_offset": 0.3,
    "margin_offset_axis": 0.9,
    "grad_fac": 0.9,
    "bias_fac": 1.0,
    "bias_fac_min": 0.3,
    "normalize": True,
}
data_gen = sample_ellipses  # data generator function

# ----- data set configuration -----
set_params = {
    "num_train": 25000,
    "num_val": 1000,
    "num_test": 1000,
    "path": DATA_PATH,
}
