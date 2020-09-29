import numpy as np


def change_base(image, eigvecs, mean_image):
    aux = image - mean_image
    inv_eigvecs = np.linalg.pinv(eigvecs)
    W = np.dot(inv_eigvecs, aux)
    return W
