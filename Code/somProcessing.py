import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import imgaug.augmenters as iaa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from EncoderDecoder import *
from minisom import MiniSom
from somImplement import MiniSomImplement
#from tensorflow.keras.callbacks import EarlyStopping
from EncoderDecoder import *

train_dataset = 'Dataset2/asl_alphabet_train/asl_alphabet_train/'
test_dataset = 'Dataset2/asl_alphabet_test/asl_alphabet_test'

train_folder = os.listdir(train_dataset)
test_folder = os.listdir(test_dataset)
input_images = []
target_images = []
labels= []
model,encoder = getEncoderDecoderWithTarget()

def preprocess(image_path,image_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = gray_img / 255.0
    return img_norm

for folder in train_folder:
    #print(folder)
    alphabet_folder = os.path.join(train_dataset, folder)
    if alphabet_folder.__contains__('.DS_Store'):
        continue
    images = os.listdir(alphabet_folder)
    target = alphabet_folder.split('/')[-1]
    cnt = 0
    for image in images:
        image_path = os.path.join(alphabet_folder,image)
        if image_path.__contains__('.DS_Store'):
            continue
        print(image_path)
        normalized= preprocess(image_path,image_size=(200,200))
        new_dim = encoder.predict(np.array([normalized.reshape(200,200)]))
        #model.predict(np.array([x_train[3].reshape(200,200)]))
        #input_images.append(normalized)
        input_images.append(new_dim)
        labels.append(target)
        cnt+=1
        if cnt >250:
            break
test_img = []
test_labels = []
for folder in test_folder:
    #print(folder)
    img_folder = os.path.join(test_dataset, folder)
    if img_folder.__contains__('.DS_Store'):
        continue
    print(img_folder)
    #images = os.listdir(alphabet_folder)
    target = img_folder.split('/')[-1].split('_')[0]
    print(target)
    if img_folder.__contains__('.DS_Store'):
            continue
    print(img_folder)
    normalized= preprocess(img_folder,image_size=(200,200))
    new_dim = encoder.predict(np.array([normalized.reshape(200,200)]))
        #model.predict(np.array([x_train[3].reshape(200,200)]))
        #input_images.append(normalized)
    test_img.append(new_dim)
    test_labels.append(target)
        # cnt+=1
        # if cnt >250:
        #     break


test_data = np.array(input_images).copy()
test_data = test_data.reshape((len(test_data),256))
# import pickle
# with open('test_data.pickle', 'wb') as f:
#     # Use pickle.dump() to save the list to the file
#     pickle.dump(test_data, f)
#
# with open('test_labels.pickle', 'wb') as f:
#     # Use pickle.dump() to save the list to the file
#     pickle.dump(test_labels, f)

data = np.array(input_images).copy()
data = data.reshape((len(data),256))
# n_images = data.shape[0]
# image_size = data.shape[1]
# n_neurons = 10
# m_neurons = 10
# som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5,
#               neighborhood_function='gaussian', random_seed=0)
# # Train the SOM with the data
# som.train_random(data, 1000)

# # Plot the SOM
# som.plot_map()
# import pickle
# with open('data_list.pickle', 'wb') as f:
#     # Use pickle.dump() to save the list to the file
#     pickle.dump(data, f)
#
# with open('labels_list.pickle', 'wb') as f:
#     # Use pickle.dump() to save the list to the file
#     pickle.dump(labels, f)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Fit and transform the labels
encoded_labels = le.fit_transform(labels)



# Print the encoded labels
# print(encoded_labels)
# Initialize the MiniSom
grid_size = (50, 50) # Change to suit your needs
input_len = 256 # Change to match the number of features in your data
sigma = 2.0 # Change to adjust the neighborhood radius
learning_rate = 0.5 # Change to adjust the learning rate
iterations = 10000 # Change to adjust the number of iterations
som = MiniSom(grid_size[0], grid_size[1], input_len, sigma=sigma, learning_rate=learning_rate,neighborhood_function='gaussian')
som.random_weights_init(data)
# n_neurons, m_neurons, input_len, sigma=1.0, learning_rate=0.5,
#                 neighborhood_function='gaussian', random_seed=None

#Train the MiniSom
som.train_random(data, iterations)
# from som import MiniSomImplement
som_i = MiniSomImplement(grid_size[0], grid_size[1], input_len, sigma=sigma, learning_rate=learning_rate,neighborhood_function='gaussian')
som_i.train(data, iterations)
som_i.plot_weights_target(encoded_labels,data)
max_iter = 10000
q_error = []
t_error = []

for i in range(max_iter):
    rand_i = np.random.randint(len(data))
    som.update(data[rand_i], som.winner(data[rand_i]), i, max_iter)
    q_error.append(som.quantization_error(data))
    t_error.append(som.topographic_error(data))

plt.plot(np.arange(max_iter), q_error, label='quantization error')
plt.plot(np.arange(max_iter), t_error, label='topographic error')
plt.ylabel('error')
plt.xlabel('iteration index')
plt.legend()
plt.show()
# # Visualize the results
# plt.figure(figsize=(16, 16))
# for i, (x, t) in enumerate(zip(data, encoded_labels)):
#     w = som_i.find_winner(x)
#     print(w)
#     plt.text(w[0]+.5, w[1]+.5, str(t), color=plt.cm.tab20(t / 10.), fontdict={'weight': 'bold',  'size': 11})
# plt.axis([0, grid_size[0], 0, grid_size[1]])
# plt.title('SOM Clusters')
# plt.show()



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

for i, (x, t) in enumerate(zip(data, encoded_labels)):
     w = som.winner(x)
     ax.scatter(w[0], w[1], t, color=plt.cm.tab20(t / 10.), s=100, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Encoded Label')
ax.set_title('SOM Clusters in 3D')
plt.show()




