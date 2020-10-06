import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from utils import eig


def KPCA(images, anon, eigenvectors, degree):
    images_quantity = len(images)

    # Kernel matrix
    K = (np.dot(images, images.T) / images_quantity + 1) ** degree

    # Centering
    uno_matrix = np.ones([images_quantity, images_quantity]) / images_quantity
    K = K - np.dot(uno_matrix, K) - np.dot(K, uno_matrix) + np.dot(uno_matrix, np.dot(K, uno_matrix))

    # Eigenvalues and eigenvectors
    eigenvals, eigenvec = eig(K)

    # Reorder eigenvalues
    # lambdas = np.flipud(lambdas)
    # alpha = np.fliplr(alpha)

    for col in range(eigenvec.shape[1]):
        eigenvec[:, col] = eigenvec[:, col] / eigenvals[col] # / np.sqrt(abs(eigenvals[col]))
        # why divide?

    # Projection
    images_projection = np.dot(K.T, eigenvec)
    uno_matrix_anon = np.ones([1, images_quantity]) / images_quantity
    Kanon = (np.dot(anon, images.T) / images_quantity + 1) ** degree
    Kanon = Kanon - np.dot(uno_matrix_anon, K) - np.dot(Kanon, uno_matrix) + np.dot(uno_matrix_anon, np.dot(K, uno_matrix))
    anon_projection = np.dot(Kanon, eigenvec)

    # Reduced projeccion
    return images_projection[:, 0:eigenvectors], anon_projection[:, 0:eigenvectors]


    # return proyec

    # svm

    # nmax = alpha.shape[1]
    # nmax = 100
    # accs = np.zeros([nmax, 1])
    # for neigen in range(1, nmax):
    #     # Me quedo sólo con las primeras autocaras
    #     # proyecto
    #     improy = improypre[:, 0:neigen]
    #     imtstproy = imtstproypre[:, 0:neigen]

    #     # SVM
    #     # entreno
    #     clf = svm.LinearSVC()
    #     clf.fit(improy, person.ravel())
    #     accs[neigen] = clf.score(imtstproy, persontst.ravel())
    #     print('Precisión con {0} autocaras: {1} %\n'.format(neigen, accs[neigen] * 100))

    # fig, axes = plt.subplots(1, 1)
    # axes.semilogy(range(nmax), (1 - accs) * 100)
    # axes.set_xlabel('No. autocaras')
    # axes.grid(which='Both')
    # fig.suptitle('Error')
    # fig.show()
