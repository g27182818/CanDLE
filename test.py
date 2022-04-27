from unicodedata import category
import pandas as pd
import numpy as np
import os
import time
from utils import *

start = time.time()
matrix_data = pd.read_feather(os.path.join("data", "final_toil", "data_matrix.feather"))
categories = pd.read_csv(os.path.join("data", "final_toil", "categories.csv"), encoding = "cp1252")
phenotypes = pd.read_csv(os.path.join("data", "final_toil", "phenotypes.csv"), encoding = "cp1252")
end = time.time()

print("Time to load data: {}".format(end - start))

breakpoint()
