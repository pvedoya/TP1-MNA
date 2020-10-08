import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from utils import eig


def KPCA(images, anon, eigenvector_cap, degree):
    images_quantity = len(images)

    # Kernel matrix
    K = (np.dot(images, images.T) / images_quantity + 1) ** degree

    # Centering
    uno_matrix = np.ones([images_quantity, images_quantity]) / images_quantity
    K = K - np.dot(uno_matrix, K) - np.dot(K, uno_matrix) + np.dot(uno_matrix, np.dot(K, uno_matrix))

    # Eigenvalues and eigenvectors
    eigenvals, eigenvec = eig(K)

    # TODO Reorder eigenvalues

    for col in range(eigenvec.shape[1]):
        eigenvec[:, col] = eigenvec[:, col] / eigenvals[col]

    # Projection
    images_projection = np.dot(K.T, eigenvec)
    uno_matrix_anon = np.ones([1, images_quantity]) / images_quantity
    Kanon = (np.dot(anon, images.T) / images_quantity + 1) ** degree
    Kanon = Kanon - np.dot(uno_matrix_anon, K) - np.dot(Kanon, uno_matrix) + np.dot(uno_matrix_anon, np.dot(K, uno_matrix))
    anon_projection = np.dot(Kanon, eigenvec)

    # Reduced projeccion
    eigenvectors = min(eigenvector_cap, len(images_projection[0]))
    return images_projection[:, 0:eigenvectors], anon_projection[:, 0:eigenvectors]
