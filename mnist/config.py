import os

from data_management import MNISTData
from operators import Gaussian


DATA_PATH = os.path.join("raw_data")
RESULTS_PATH = os.path.join("results")

# ----- random seeds -----
torch_seed = 1
numpy_seed = 2
matrix_seed = 3

# ----- signal configuration -----
n = 784  # signal dimension
data_params = {}  # additional data generation parameters
data_gen = MNISTData(DATA_PATH)  # data generator function

# ----- measurement configuration -----
m = 300  # measurement dimension
meas_params = {"seed": matrix_seed}  # additional measurement parameters
meas_op = Gaussian  # measurement operator

# ----- data set configuration -----
set_params = {
    "num_train": 60000,
    "num_val": 5000,
    "num_test": 5000,
    "path": DATA_PATH,
}
