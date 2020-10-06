import numpy as np
from os import listdir
from os.path import join, isdir
from typing import Tuple
import matplotlib.pyplot as plt
from numpy import linalg


def change_base(image, eigvecs, mean_image):
    aux = np.transpose(np.transpose(image) - np.transpose(mean_image))
    inv_eigvecs = np.linalg.pinv(eigvecs)
    W = np.dot(inv_eigvecs, aux)
    return W


def find_closest_match(vector, matrix):
    matchPercentages = np.zeros(len(matrix))
    i = 0
    for v in matrix:
        matchPercentages[i] = np.linalg.norm(v - vector)
        i += 1

    minDistance = np.amin(matchPercentages)
    matchPosition = np.where(matchPercentages == minDistance)[0][0]

    return matrix[matchPosition], matchPosition, minDistance


def generate_photo_matrix(photo_set_path, height, width, people_amount, per_person_amount):
    photo_area = height * width
    people_area = people_amount * per_person_amount

    dirs = [f for f in listdir(photo_set_path)
            if isdir(join(photo_set_path, f))]
    photo_matrix = np.zeros([people_area, photo_area])

    img_num = 1
    for person in dirs:
        for photo in listdir(photo_set_path + '/' + person):
            photo_path = photo_set_path + '/' + person + '/' + photo
            photo_matrix[img_num-1,
                         :] = generate_photo_vector(photo_path, height, width)
            img_num += 1
            if (img_num-1) % per_person_amount == 0:
                break

    return photo_matrix


def generate_photo_vector(photo_path, height, width):
    photo_area = height * width
    photo = plt.imread(photo_path)
    return np.reshape(photo, [1, photo_area])


def generate_face(vector, height, width, path):
    face = vector.reshape(height, width)
    plt.imsave(path, face, cmap='gray')


def power_method(A: np.ndarray, x: np.ndarray = 'none', tolerance: float = 1e-10) -> Tuple[float, np.ndarray]:
    rows = A.shape[0]
    cols = A.shape[1]

    if not rows == cols:
        raise Exception('Matrix is not square')

    A_copy = np.copy(A)

    eig_value = 0
    eig_vector = x if x != 'none' else np.random.rand(cols, 1)

    while True:
        eig_vector_old = eig_vector

        eig_value = np.linalg.norm(A_copy.dot(eig_vector))
        eig_vector = A_copy.dot(eig_vector)/eig_value

        delta = abs(eig_vector) - abs(eig_vector_old)
        if abs(delta).all() < tolerance and np.linalg.norm(delta, 2) < tolerance:
            break
    return eig_value, eig_vector


# def rayleigh(A: np.ndarray, x: np.ndarray) -> float:
#     xt = np.matrix.transpose(x)
#     norm_22 = np.linalg.norm(x, 2) ** 2
#     ret1 = xt.dot(A)
#     ret = ret1.dot(x)/norm_22
#     return ret[0][0]

# def inv_power_method(A: np.ndarray, x: np.ndarray = 'none', tolerance: float = 1e-10) -> Tuple[float, np.ndarray]:
#     _w, v = power_method(np.linalg.inv(A), x, tolerance)
#     return rayleigh(A, v), v

