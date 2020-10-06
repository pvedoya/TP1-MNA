from argparse import ArgumentParser
import configparser
import numpy as np
from utils import change_base
from utils import find_closest_match
from utils import generate_photo_matrix
from utils import generate_photo_vector
from utils import generate_face
from utils import calculate_match_percentages
from KPCA import KPCA
from PCA import PCA
from face_detection import face_recognition
# Reader of the ini file, read fnc receives the path
config = configparser.ConfigParser()
config.read('configuration.ini')

is_kpca = config.getboolean('SETTINGS', 'IS_KPCA')
kpca_degree = config.getint('SETTINGS', 'KPCA_DEGREE')
photo_set_path = config.get('SETTINGS', 'PHOTO_SET')
is_video = config.getboolean('SETTINGS', 'IS_VIDEO')


photo_height = config.getint('IMAGES_DATA', 'HEIGHT')
photo_width = config.getint('IMAGES_DATA', 'WIDTH')
people_amount = config.getint('IMAGES_DATA', 'PEOPLE_PER_SET')
per_person_amount = config.getint('IMAGES_DATA', 'IMG_PER_PERSON')

eigenvector_amount = config.getint('RESULTS_DATA', 'EIGENVECTORS')
anon_photo_path = config.get('RESULTS_DATA', 'PHOTO')
threshold = config.getfloat('RESULTS_DATA', 'THRESHOLD')
# Creates matrix of shape (n_photos, n_measurements),
# dictionary of photo_row -> photo_path,
# array that matches each photo to a person id
# and dictionary of person_id -> person_name
photo_matrix, photo_dict, people_groups, people_dict = generate_photo_matrix(photo_set_path, photo_height, photo_width, people_amount,
                                                 per_person_amount)

# Creates array of anonymous photo to analize
if is_video:
    print('Opening webcam viewer...')
    gray_array = face_recognition(name='person', pictures=1)
    print('Face captured. Running facial recognition...')
    anon_vector = np.reshape(gray_array, [1, photo_height*photo_width])
else:
    anon_vector = generate_photo_vector(anon_photo_path, photo_height, photo_width)

if is_kpca:
    weights, anon_weight = KPCA(photo_matrix, anon_vector, eigenvector_amount, kpca_degree)
else:
    weights, anon_weight = PCA(photo_matrix, anon_vector, eigenvector_amount)

# Returns array of match percentages for each person
match_percentages = calculate_match_percentages(weights, people_groups, anon_weight)
sorted_indexes = np.argsort(match_percentages)

print('Match percentages:')
for idx in sorted_indexes:
    percentage = match_percentages[idx]
    print(f'Person: {people_dict[idx+1]} - match: {match_percentages[idx]*100}%')
