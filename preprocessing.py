import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_input(views, likes, comments, sentiment):
    """
    Convert inputs into model-ready normalized format.
    """
    data = np.array([[views, likes, comments, sentiment]])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled
