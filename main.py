from argparse import ArgumentParser
import configparser
import numpy as np
from utils import change_base
from utils import find_closest_match
from utils import generate_photo_matrix
from utils import generate_photo_vector
from utils import generate_face
from utils import calculate_match
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
svm_c = config.getfloat('SETTINGS', 'SVM_C')

photo_height = config.getint('IMAGES_DATA', 'HEIGHT')
photo_width = config.getint('IMAGES_DATA', 'WIDTH')
people_amount = config.getint('IMAGES_DATA', 'PEOPLE_PER_SET')
per_person_amount = config.getint('IMAGES_DATA', 'IMG_PER_PERSON')

eigenvector_amount = config.getint('RESULTS_DATA', 'EIGENVECTORS')
anon_photo_path = config.get('RESULTS_DATA', 'PHOTO')

# Creates matrix of shape (n_photos, n_measurements),
# dictionary of photo_row -> photo_path,
# array that matches each photo to a person id
# and dictionary of person_id -> person_name
photo_matrix, people_groups, people_dict = generate_photo_matrix(photo_set_path, photo_height, photo_width, people_amount,
                                                 per_person_amount)

# Creates array of anonymous photo to analize
if is_video:
    print('Opening webcam viewer...')
    gray_array = face_recognition(photo_width, photo_height, name='person', pictures=1)
    print('Face captured. Running facial recognition...')
    anon_vector = np.reshape(gray_array, [1, photo_height*photo_width])
else:
    anon_vector = generate_photo_vector(anon_photo_path, photo_height, photo_width)

if is_kpca:
    weights, anon_weight = KPCA(photo_matrix, anon_vector, eigenvector_amount, kpca_degree)
else:
    weights, anon_weight = PCA(photo_matrix, anon_vector, eigenvector_amount)

# Returns id of matched person
match_id = calculate_match(weights, people_groups, anon_weight, svm_c)
group_id_list = list(map(int, dict.fromkeys(people_groups)))
match_dict = {}
for g in group_id_list:
    i = g-1
    group_eigenvalues = weights[i*per_person_amount:i*per_person_amount+per_person_amount, :]
    mean_group_eigenvalues = np.mean(group_eigenvalues, 0).reshape(((weights.shape[1],1)))
    p = np.absolute(np.dot(anon_weight, mean_group_eigenvalues))/(np.linalg.norm(anon_weight, 2) * np.linalg.norm(mean_group_eigenvalues, 2))
    match_dict[people_dict[g]] = (p[0][0]*100)

for k,v in sorted(match_dict.items(), key=lambda x: x[1], reverse=True):
    print("%-20s %5.2f%%"%(f"Person: {k}", v))

print(f'SVM Match: {people_dict[match_id]}')