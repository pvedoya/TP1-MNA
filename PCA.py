from os import listdir
from os.path import join, isdir
import numpy as np
import matplotlib.pyplot as plt
from utils import change_base
from utils import eigenvectors
from utils import qr_householder


m = [[1,0,0,2,1,2],[12,0,3,2,4,2],[31,4,0,42,21,2],[61,50,0,32,1,12],[1,50,30,2,11,32],[15,1,0,25,13,24]]

# [eigvals, eigvecs] = np.linalg.eig(m)

# print(eigvecs)

# [aux1,aux2] = eigenvectors(m)

# print(aux2)

[Q, R] = np.linalg.qr(m)
print(Q, "\n\n\n", R)
print("\n\n\n\n")

[Q, R] = qr_householder(m)
print(Q, "\n\n\n", R)



def PCA(images, height, width):

    ######################TODO:deberia recibir todos los datos q estan entre los ## desde main
    # hay que pasarle la matriz de fotos, y el image_test que es el set de testeo
    # FILES
    # path = 'att_faces/'
    # dirs = [f for f in listdir(path) if isdir(join(path, f))]
    #
    # # IMAGE DATA
    # height = 92
    # width = 112
    # image_area = height * width
    #
    # person_pool = 40
    # training_amount = 6
    # test_amount = 4
    # training_size = person_pool * training_amount
    # test_size = person_pool * test_amount
    #
    # # TRAINING SET
    # images = np.zeros([training_size, image_area])
    # person = np.zeros([training_size, 1])
    # img_n = 0
    # people = 0
    #
    # for directory in dirs:
    #     for k in range(1, training_amount + 1):
    #         a = plt.imread(path + directory + '/{}'.format(k) + '.pgm') / 255.0
    #         images[img_n, :] = np.reshape(a, [1, image_area])
    #         person[img_n, 0] = people
    #         img_n += 1
    #     people += 1

    # # TEST SET
    # image_test = np.zeros([test_size, image_area])
    # person_test = np.zeros([test_size, 1])
    # img_n = 0
    # people = 0
    #
    # for directory in dirs:
    #     for k in range(training_amount, 10):
    #         a = plt.imread(path + directory + '/{}'.format(k) + '.pgm') / 255.0
    #         image_test[img_n, :] = np.reshape(a, [1, image_area])
    #         person_test[img_n, 0] = people
    #         img_n += 1
    #     people += 1

    #####################################

    # MEAN FACE
    mean_image = np.mean(images, 0)
    # fig, axes = plt.subplots(1, 1)
    # axes.imshow(np.reshape(mean_image, [width, height]) * 255, cmap='gray')
    # fig.suptitle('MEAN IMAGE')

    # SUBTRACT THE MEAN
    images = [images[k, :] - mean_image for k in range(images.shape[0])]
    # image_test = [image_test[k, :] - mean_image for k in range(image_test.shape[0])]

    # PCA
    # U, S, V = np.linalg.svd(images, full_matrices=False)   # esto es para comparar errores

    C_reduced = (1 / (len(images) - 1)) * (np.dot(images, np.transpose(images)))
    [eigvals, eigvecs] = eigenvectors(C_reduced)

    C_eigvecs = np.dot(np.transpose(images), eigvecs)

    W = change_base(np.transpose(images), C_eigvecs, mean_image)

    return [C_eigvecs, np.transpose(W), np.reshape(mean_image, [1, len(mean_image)])]
