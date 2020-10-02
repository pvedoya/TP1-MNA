import numpy as np
from os import listdir
from os.path import join, isdir
import matplotlib.pyplot as plt

ERROR = 0.0001    # TODO:define a proper value for the error


def change_base(image, eigvecs, mean_image):
    aux = np.transpose(np.transpose(image) - np.transpose(mean_image))
    inv_eigvecs = np.linalg.pinv(eigvecs)
    W = np.dot(inv_eigvecs, aux)
    return W


# https://byumcl.bitbucket.io/bootcamp2014/_downloads/Lab13v1.pdf, complemented with
# https://stackoverflow.com/questions/60956390/performing-householder-reflection-of-a-vector-for-qr-decomposition
def qr_householder(A):
    A = np.array(A)
    m, n = A.shape
    Q = np.eye(m)
    R = np.copy(A)

    k = min(m-1, n)

    for j in range(k):
        I = np.eye(m)
        vec = R[j:, j]
        e = np.zeros_like(vec)
        e[0] = np.linalg.norm(vec) * np.sign(A[j, j])
        vec = vec + e
        aux = vec / np.linalg.norm(vec)
        I[j:, j:] -= 2.0 * np.outer(aux, aux)
        R = np.dot(I, R)
        Q = np.dot(I, Q)
    return np.transpose(Q), R


def eigenvectors(A):
    i = 0
    aux_eig_val = np.diag(A)  # aproximación inicial de autovalores
    q, r = qr_householder(A)
    eig_vec = q
    a = np.matmul(r, q)
    eig_val = np.diag(a)

    while i < 50:
        q, r = qr_householder(a)
        a = np.matmul(r, q)
        eig_vec = np.matmul(eig_vec, q)
        aux_eig_val = eig_val
        eig_val = np.diag(a)
        if np.linalg.norm(np.subtract(eig_val, aux_eig_val)) < ERROR:
            break
        i = i + 1

    sorting = np.argsort(eig_val)[::-1]
    eig_val = eig_val[sorting]
    eig_vec = eig_vec[sorting]
    return eig_val, eig_vec


def find_closest_match(vector, matrix):
    matchPercentages = np.zeros(len(matrix))
    i = 0
    for v in matrix:
        matchPercentages[i] = np.linalg.norm(v - vector)
        i += 1

    minDistance = np.amin(matchPercentages)
    matchPosition = np.where(matchPercentages == minDistance)[0][0]

    return matrix[matchPosition], matchPosition, minDistance


# TODO
def generate_photo_matrix(photo_set_path, height, width, people_amount, per_person_amount):
    photo_area = height * width
    people_area = people_amount * per_person_amount

    dirs = [f for f in listdir(photo_set_path) if isdir(join(photo_set_path, f))]
    photo_matrix = np.zeros([people_area, photo_area])

    img_num = 1
    for person in dirs:
        for photo in listdir(photo_set_path + '/' + person):
            photo_path = photo_set_path + '/' + person + '/' + photo
            photo_matrix[img_num-1, :] = generate_photo_vector(photo_path, height, width)
            img_num += 1
            if (img_num-1) % per_person_amount == 0:
                break

    return photo_matrix


# TODO
def generate_photo_vector(anon_photo_path, height, width):
    photo_area = height * width
    photo = plt.imread(anon_photo_path) / 255.0
    return np.reshape(photo, [1, photo_area])
