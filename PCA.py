from os import listdir
from os.path import join, isdir
import numpy as np
from utils import change_base
from utils import eig
from sklearn.preprocessing import StandardScaler

DEC = 7


def PCA(images, anon_vector, eigenvectors_cap):

    # Mean Face
    mean_image = np.mean(images, 0)

    # Dispersion
    disp = np.sqrt(np.var(images, 0)*len(images)/(len(images)-1))

    # Standardize
    images_og = images
    images = np.array([np.divide(images[k, :] - mean_image, disp) for k in range(images.shape[0])])

    # Reduced Covariance Matrix
    C_reduced = (1 / (len(images) - 1)) * (np.dot(images, np.transpose(images)))
    [eigvals, eigvecs] = eig(C_reduced)

    # Round eigenvectors and eigenvalues to DEC decimals, so as to have well defined 0s
    eigvals = np.round(eigvals, decimals=DEC)
    for i in range(0, len(eigvecs)):
        eigvecs[i] = np.round(eigvecs[i], decimals=DEC)

    # Keep only eigenvectors of big eigenvalues
    i = 0
    big_eigvecs = []
    for eigval in eigvals:
        if eigval != 0:
            big_eigvecs.append(eigvecs[:, i])
        i += 1
    big_eigvecs = np.transpose(big_eigvecs)

    # Calculate covariance eigenvectors
    C_eigvecs = np.dot(np.transpose(images), big_eigvecs)

    # Normalize
    C_eigvecs = np.transpose(np.array([C_eigvecs[:, k] / np.linalg.norm(C_eigvecs[:, k])
                                       for k in range(len(C_eigvecs[0]))]))

    mean_image = np.reshape(mean_image, [len(mean_image), 1])

    # Weights of photo matrix on base of covariance eigenvectors
    W = change_base(np.transpose(images_og), C_eigvecs, mean_image)

    # Standardize weights
    scaler = StandardScaler()
    scaled_weights = scaler.fit_transform(W)
    W = scaled_weights

    # Weights of anonymous photo
    anon_weight = change_base(np.reshape(anon_vector, [len(anon_vector[0]), 1]), C_eigvecs, mean_image)

    eigenvector_amount = min(eigenvectors_cap, len(W[0]))
    return np.transpose(W)[:, 0:eigenvector_amount], np.reshape(anon_weight, [1, len(anon_weight)])[:, 0:eigenvector_amount]
