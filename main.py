from argparse import ArgumentParser
import numpy as np
from utils import change_base
from utils import find_closest_match
from utils import generate_photo_matrix
from utils import generate_photo_vector
from KPCA import calculate_kpca as KPCA
from PCA import PCA

ap = ArgumentParser()

ap.add_argument("-s", "--photoset", required=True, help="Photo set for analysis")
ap.add_argument("-H", "--height", required=True, help="Photos height")
ap.add_argument("-W", "--width", required=True, help="Photos width")
ap.add_argument("-P", "--people", required=True, help="Amount of people in photo set")
ap.add_argument("-A", "--amount", required=True, help="Amount of photos each person has")
ap.add_argument("-e", "--eigenvectors", required=False, help="Amount of eigenvectors to use", default=90)
ap.add_argument("-k", "--kpca", required=False, help="Calculate KPCA instead (default is PCA)", action='store_true')
ap.add_argument("-p", "--photo", required=True, help="Path of photo to analyze")
ap.add_argument("-t", "--threshold", required=True, help="Minimum threshold (0-100) for match", default="77")

args = ap.parse_args()

eigenvector_amount = int(args.eigenvectors)
is_kpca = args.kpca
photo_set_path = args.photoset
anon_photo_path = args.photo
threshold = float(args.threshold)
photo_height = int(args.height)
photo_width = int(args.width)
people_amount = int(args.people)
per_person_amount = int(args.amount)

photo_matrix = generate_photo_matrix(photo_set_path, photo_height, photo_width, people_amount, per_person_amount)
print(f'photo matrix dimensions: {photo_matrix.shape}')
anon_vector = generate_photo_vector(anon_photo_path, photo_height, photo_width)
print(f'photo vector dimensions: {anon_vector.shape}')

# photo_matrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1]])
# anon_vector = np.reshape([0, 1, 0, 0, 0, 0], [6, 1])
# photo_height = 3
# photo_width = 2

# print(photo_matrix)

if is_kpca:
    eigenvectors, weights, meanPhoto = KPCA(photo_matrix, photo_height, photo_width)  # TODO
else:
    eigenvectors, weights, meanPhoto = PCA(photo_matrix, photo_height, photo_width)

# print(f'weights: \n{weights}\neigenv:\n{eigenvectors}\nmean photo: \n{meanPhoto}')

anonWeight = change_base(np.reshape(anon_vector, [len(anon_vector[0]), 1]), eigenvectors,
                         np.reshape(meanPhoto, [len(meanPhoto[0]), 1]))
anonWeight = np.reshape(anonWeight, [1, len(anonWeight)])

closestVector, row, match_percentage = find_closest_match(anonWeight, weights)

print(f'Closest match: {row}, distance: {match_percentage}')

if match_percentage > threshold:
    print(f'No match found. Photo had match distance of {match_percentage}, '
          f'higher than threshold of {threshold}')
else:
    print(f'Match found with {match_percentage} distance.')
    original_photo = photo_matrix[row]
    print(f'Photo Number ID: {row}')
    # print_match(original_photo)
