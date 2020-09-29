from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
# original  dataset from https://cs.nyu.edu/~roweis/data.html

imgs = loadmat('olivettifaces.mat')
faces = imgs['faces']


for img in range(400):
  face = faces[:, img].reshape(64, 64)
  plt.imsave('./images/{img}.png'.format(img=img), np.rot90(face, axes=(1, 0)), cmap='gray')


