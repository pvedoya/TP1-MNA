from argparse import ArgumentParser
import configparser
import numpy as np
from utils import change_base
from utils import find_closest_match
from utils import generate_photo_matrix
from utils import generate_photo_vector
from utils import generate_face
from KPCA import calculate_kpca as KPCA
from PCA import PCA

# Reader of the ini file, read fnc receives the path
config = configparser.ConfigParser()
config.read('configuration.ini')

is_kpca = config.getboolean('SETTINGS', 'IS_KPCA')
photo_set_path = config.get('SETTINGS', 'PHOTO_SET')

photo_height = config.getint('IMAGES_DATA', 'HEIGHT')
photo_width = config.getint('IMAGES_DATA', 'WIDTH')
people_amount = config.getint('IMAGES_DATA', 'PEOPLE_PER_SET')
per_person_amount = config.getint('IMAGES_DATA', 'IMG_PER_PERSON')

eigenvector_amount = config.getint('RESULTS_DATA', 'EIGENVECTORS')
anon_photo_path = config.get('RESULTS_DATA', 'PHOTO')
threshold = config.getfloat('RESULTS_DATA', 'THRESHOLD')

print(f'Using database of {per_person_amount} photos of each {people_amount} people')
print(f'Analizing photo \'{anon_photo_path}\'')

photo_matrix, photo_dict = generate_photo_matrix(photo_set_path, photo_height, photo_width, people_amount,
                                                 per_person_amount)
photos_og = np.copy(photo_matrix)
anon_vector = generate_photo_vector(anon_photo_path, photo_height, photo_width)

if is_kpca:
    eigenvectors, weights, meanPhoto = KPCA(photo_matrix)  # TODO
else:
    eigenvectors, weights, meanPhoto = PCA(photo_matrix)

anonWeight = change_base(np.reshape(anon_vector, [len(anon_vector[0]), 1]), eigenvectors,
                         np.reshape(meanPhoto, [len(meanPhoto[0]), 1]))
anonWeight = np.reshape(anonWeight, [1, len(anonWeight)])

closestVector, row, match_percentage = find_closest_match(anonWeight, weights)

if match_percentage > threshold:
    print(f'No match found. Photo had match distance of {match_percentage}, '
          f'higher than threshold of {threshold}')
    print(f'Closest match: {photo_dict[row]}, distance: {match_percentage}')
else:
    print(f'Match found with {match_percentage} distance.')
    print(f'Photo Match Path: {photo_dict[row]}')
