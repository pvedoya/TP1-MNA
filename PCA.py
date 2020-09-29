import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from os import listdir
from os.path import join, isdir
import configparser

# Files
path = "att_faces/"
dirs = [f for f in listdir(path) if isdir(join(path, f))]

# Images data
img_width = 92
img_height = 112
img_area = img_width * img_height
images_per_person = 5  # nose

training_number = 6
test_number = 4
person_pool = 40
training_size = training_number * person_pool
test_size = test_number * person_pool

# SET BUILDING
training_set = np.zeros([training_size, img_area])
person_training = np.zeros([training_size, 1])
images = 0
people = 0

for dir in dirs:
    for i in range(1, training_number+1):
        a = plt.imread(path + dir + '/{}'.format(i) + '.pgm') / 255.0
        training_set[images, :] = np.reshape(a, [1, img_area])
        person_training[images, 0] = people
        images += 1
    people += 1

testing_set = np.zeros([test_size, img_area])
person_testing = np.zeros([test_size, 1])
images = 0
people = 0

for dir in dirs:
    for k in range(test_number, 10):
        a = plt.imread(path + dir + '/{}'.format(k) + '.pgm')/255.0
        testing_set[images, :] = np.reshape(a, [1, img_area])
        person_testing[images, 0] = people
        images += 1
    people += 1

print(person_training)
print(person_testing)