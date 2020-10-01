from argparse import ArgumentParser
from utils import change_base
from utils import find_closest_match
from utils import generate_photo_matrix
from utils import generate_photo_vector
from PCA import PCA
from KPCA import calculate_kpca as KPCA

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

eigenvector_amount = args.eigenvectors
is_kpca = args.kpca
photo_set_path = args.photoset
anon_photo_path = args.photo
threshold = args.threshold
photo_height = args.height
photo_width = args.width
people_amount = args.people
per_person_amount = args.amount

photo_matrix = generate_photo_matrix(photo_set_path, photo_height, photo_width, people_amount, per_person_amount)
anon_vector = generate_photo_vector(anon_photo_path, photo_height, photo_width)

if is_kpca:
    weights, eigenvectors, meanPhoto = KPCA(photo_matrix, photo_height, photo_width)  # TODO
else:
    weights, eigenvectors, meanPhoto = PCA(photo_matrix, photo_height, photo_width)

anonWeight = change_base(anon_vector, eigenvectors, meanPhoto)

closestVector, row, match_percentage = find_closest_match(anonWeight, weights)

if match_percentage < threshold:
    print(f'No match found. Photo had match percentage of {match_percentage}, '
          f'lower than threshold of {threshold}')
else:
    print(f'Match found with {match_percentage} probability.')
    original_photo = photo_matrix[row]
    print(f'Photo Number ID: {row}')
    # print_match(original_photo)
