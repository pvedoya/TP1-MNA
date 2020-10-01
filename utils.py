import numpy as np
from os import listdir
from os.path import join, isdir
import matplotlib.pyplot as plt


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
