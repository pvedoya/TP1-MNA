"""
4 parameters:
. Photo Dataset
. PCA or KPCA
. Eigenvectors amount
. Photo for analysis
. Match Threshold

photoSetPath = 'photosDirec'
calculatePCA = true
eigenAmount = 30
anonPhoto = 'anonPath'
matchThreshold = 85

photoMatrix = generatePhotoMatrix(photosSetPath)
anonVector = generatePhotoVector(anonPhoto)

if calculatePCA:
    weights, eigenvectors, meanPhoto = calculatePCA(photoMatrix)
else:
    weights, eigenvectors, meanPhoto = calculateKPCA(photoMatrix)

anonWeight = changeBase(anonVector, eigenvectors, meanPhoto)

closestVector, matchPercentage = findClosestMatch(anonWeight, weights)

if matchPercentage < threshold:
    print(f'No match found. Photo had match percentage of {matchPercentage}, lower than threshold of {matchThreshold}')
else:
    print(f'Match found with {matchPercentage} probability.')
    originalPhoto = findOriginalPhoto(closestVector, weights, photoMatrix)
    printMatch(originalPhoto)

"""

