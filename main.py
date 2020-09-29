import argparse


ap = argparse.ArgumentParser()

ap.add_argument("-s", "--photoset", required=True, help="Photo set for analysis")
ap.add_argument("-e", "--eigenvectors", required=False, help="Amount of eigenvectors to use", default=90)
ap.add_argument("-k", "--kpca", required=False, help="Calculate KPCA instead (default is PCA)", action='store_true')
ap.add_argument("-p", "--photo", required=True)
ap.add_argument("-t", "--threshold", required=True, help="Minimum threshold (0-100) for match", default="77")

args = ap.parse_args()

eigenvector_amount = args.eigenvectors
is_kpca = args.kpca
photo_set_path = args.photoset
anon_photo = args.photo
threshold = args.photo

# photo_matrix = generate_photo_matrix(photo_set_path)
# anon_vector = generate_photo_vector(anon_photo)
#
# if is_kpca:
#     weights, eigenvectors, meanPhoto = calculate_KPCA(photo_matrix, eigenvector_amount)
# else:
#     weights, eigenvectors, meanPhoto = calculate_PCA(photo_matrix, eigenvector_amount)
#
# anonWeight = change_base(anon_vector, eigenvectors, meanPhoto)
#
# closestVector, row, matchPercentage = find_closest_match(anonWeight, weights)
#
# if matchPercentage < threshold:
#     print(f'No match found. Photo had match percentage of {match_percentage}, '
#           f'lower than threshold of {match_threshold}')
# else:
#     print(f'Match found with {matchPercentage} probability.')
#     original_photo = find_original_photo(photo_matrix, row)
#     print_match(original_photo)
