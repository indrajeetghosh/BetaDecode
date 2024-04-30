import pandas as pd
import numpy as np
from numpy import array, mean, std, dstack,argmax


def normalize(X_Raw_Data):
    result = np.copy(X_Raw_Data)
    num_features = X_Raw_Data.shape[1]  # Number of features (columns)
    for i in range(num_features):
        max_value = np.max(X_Raw_Data[:, i])
        min_value = np.min(X_Raw_Data[:, i])
        result[:, i] = (X_Raw_Data[:, i] - min_value) / (max_value - min_value)
    return result