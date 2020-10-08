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

# Returns array of match percentages for each person
match_percentages = calculate_match_percentages(weights, people_groups, anon_weight)
sorted_indexes = np.argsort(match_percentages)
# group = matched_groups[0]
match_dict = {}
for g in sorted_indexes:
    group_eigenvalues = weights[g*per_person_amount:g*per_person_amount+per_person_amount, :]
    mean_group_eigenvalues = np.mean(group_eigenvalues, 0).reshape(((weights.shape[1],1)))
    # mean_group_eigenvalues = group_eigenvalues[2, :].reshape(((175,1)))
    # print(mean_group_eigenvalues.shape)
    # print(anon_weight.shape)

    p = np.absolute(np.dot(anon_weight, mean_group_eigenvalues))/(np.linalg.norm(anon_weight, 2) * np.linalg.norm(mean_group_eigenvalues, 2))
    match_dict[people_dict[g+1]] = (p[0][0]*100)
    # print(f"Person: {people_dict[g+1]} match similarity: {p[0][0] * 100}%")
"""
Agarrar todos las pelotitas zules (es decir el grupo con el que matcheo)
mean([v1, v2, v3, v4, ...]) = [v_mean]  -> vector promedio del grupo con el que matcheo
anon_weight -> vector que estoy evaluando

para ver el porcentaje de cercania:
    p = v_mean dot anon_weight ->  un numero que esta entre 0 y norm(v_mean)*norm(anon_weight) -> por la desigualdad de cauchy shwartz
    
    p/(norm(v_mean) * norm(anon_weight)) --> un numero entro 0 y 1

    0: quiere decir que son ortogonales osea que son nada que ver
    1: quiere decir que son el mismo

    por lo tanto, podemos decir que p es el parecido entre una persona y otra


    if p < threshold:
        print('no esta en el set')

"""

for k,v in sorted(match_dict.items(), key=lambda x: x[1], reverse=True):
    print("%-20s %5.2f%%"%(f"Person: {k}", v))

print('SVM Match:')
for idx in sorted_indexes:
    percentage = match_percentages[idx]
    if match_percentages[idx] == 1:
        print(f'Person: {people_dict[idx+1]} - match: {match_percentages[idx]*100}%')