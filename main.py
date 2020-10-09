import configparser
import numpy as np
from lib.utils import generate_photo_matrix
from lib.utils import generate_photo_vector
from lib.utils import calculate_match
from lib.KPCA import KPCA
from lib.PCA import PCA
from lib.face_detection import face_recognition
import emoji

# import warnings
# warnings.filterwarnings("ignore")

confidence_kpca = {
        'very high': [65, 101],
        'high': [40, 65],
        'moderate': [25, 40],
        'low': [10, 25],
        'very low': [-1, 10]
    }

confidence_pca = {
        'very high': [90, 101],
        'high': [80, 90],
        'moderate': [70, 80],
        'low': [60, 70],
        'very low': [-1, 60]
    }

def get_confidence(v, confidence):
    emojis = {
        'very low': emoji.emojize(':sob:', use_aliases=True) + " (Very low)",
        'low': emoji.emojize(':worried:', use_aliases=True) + " (Low)",
        'moderate': emoji.emojize(':neutral_face:', use_aliases=True) + " (Moderate)",
        'high': emoji.emojize(':smiling_face_with_smiling_eyes:', use_aliases=True) + " (High)",
        'very high': emoji.emojize(':smile:', use_aliases=True) + " (Very High)"
    }

    to_return = emojis['very low']
    for key, value in confidence.items():
        if v >= value[0] and v < value[1]:
            to_return = emojis[key]
    return to_return


# Reader of the ini file, read fnc receives the path
config = configparser.ConfigParser()
config.read('configuration.ini')

is_kpca = config.getboolean('SETTINGS', 'IS_KPCA')
kpca_degree = config.getint('SETTINGS', 'KPCA_DEGREE')
photo_set_path = config.get('SETTINGS', 'PHOTO_SET')
is_video = config.getboolean('SETTINGS', 'IS_VIDEO')
svm_c = config.getfloat('SETTINGS', 'SVM_C')
svm_iter = config.getfloat('SETTINGS', 'SVM_ITER')

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
    gray_array = face_recognition(
        photo_width, photo_height, name='person', pictures=1, save_to_file=False)
    print('Face captured. Running facial recognition...')
    anon_vector = np.reshape(gray_array, [1, photo_height*photo_width])
else:
    anon_vector = generate_photo_vector(
        anon_photo_path, photo_height, photo_width)

if is_kpca:
    weights, anon_weight = KPCA(
        photo_matrix, anon_vector, eigenvector_amount, kpca_degree)

else:
    weights, anon_weight = PCA(photo_matrix, anon_vector, eigenvector_amount)
    

# Returns id of matched person
match_id = calculate_match(weights, people_groups, anon_weight, svm_c, svm_iter)
group_id_list = list(map(int, dict.fromkeys(people_groups)))
match_dict = {}
for g in group_id_list:
    i = g-1
    group_eigenvalues = weights[i*per_person_amount:i *
                                per_person_amount+per_person_amount, :]
    mean_group_eigenvalues = np.mean(
        group_eigenvalues, 0).reshape(((weights.shape[1], 1)))
    p = np.absolute(np.dot(anon_weight, mean_group_eigenvalues)) / \
        (np.linalg.norm(anon_weight, 2) * np.linalg.norm(mean_group_eigenvalues, 2))
    match_dict[people_dict[g]] = (p[0][0]*100)

# for k, v in sorted(match_dict.items(), key=lambda x: x[1], reverse=True):

print('Match found!')
similarity = match_dict[people_dict[match_id]]
confidence_dict = confidence_kpca if is_kpca else confidence_pca
print("%-15s %-12s %5.2f%%" % (f"Match: {people_dict[match_id]}", f"Confidence: {get_confidence(similarity, confidence_dict)}", similarity ))
