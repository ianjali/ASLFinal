import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import imgaug.augmenters as iaa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from EncoderDecoder import *
#from tensorflow.keras.callbacks import EarlyStopping

train_dataset = 'Dataset/asl_alphabet_train/asl_alphabet_train/'
test_dataset = 'Dataset/asl_alphabet_test/asl_alphabet_test'

train_folder = os.listdir(train_dataset)
input_images = []
target_images = []
labels= []
def preprocess(image_path,image_size):
    #print(image_path)
    #fgbg = cv2.createBackgroundSubtractorMOG2()
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    #fgmask = fgbg.apply(img)
    #cv2.imshow("title", fgmask)
    #cv2.waitKey(0)
    #target_gray = cv2.cvtColor(fgmask, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #target_norm = target_gray/255.0

    img_norm = gray_img / 255.0
    return img_norm#,target_norm

for folder in train_folder:
    #print(folder)
    alphabet_folder = os.path.join(train_dataset, folder)
    if alphabet_folder.__contains__('.DS_Store'):
        continue
    images = os.listdir(alphabet_folder)
    target = alphabet_folder.split('/')[-1]
    cnt = 0
    for image in images:
        #print(image)
        image_path = os.path.join(alphabet_folder,image)
        if image_path.__contains__('.DS_Store'):
            continue
        normalized= preprocess(image_path,image_size=(200,200))
        input_images.append(normalized)
        labels.append(target)
        cnt+=1
        if cnt == 250:
            break
        #target_images.append(target_norm)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# Fit and transform the labels
encoded_labels = le.fit_transform(labels)


#input_images = input_images
#target_images = target_images[:500]
input_images = np.array(input_images)
target_images = np.array(target_images)
target_labels = np.array(encoded_labels)

#x_train, x_test, y_train,  y_test = train_test_split(input_images, input_images, test_size=0.2, random_state=42)
x_train, x_test, y_train,  y_test = train_test_split(input_images, target_labels, test_size=0.2, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=29)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=29)
#history, model,encoder_model = trainEncoderDecoder(x_train,y_train,x_test,y_test)
#history, model, encoder_model = trainEncoderDecoderWithTarget(x_train,y_train,x_test,y_test)
#model,encoder = getEncoderDecoder()

model,encoder = getEncoderDecoderWithTarget()

# model = autoencoder_model.load_weights('autoencoder_weights_300.h5')
# encoder_model.load_weights('encoded_300.h5')

#model.save_weights('autoencoder_category.h5')
#encoder_model.save_weights('encoded_category.h5')
#decoder_model.save_weights('decoded.h5')
def predictionEncoder(x_test,y_test,model):
    predictions = []
    true =[]
    for test in x_test:
        decoded_img = model.predict(np.array([test.reshape(200, 200)]))
        pred = np.argmax(decoded_img, axis=1)
        predictions.append(pred)
        true_pred = np.argmax(y_test)
        true.append(true_pred)
    accuracy = np.mean(predictions == true)
    return accuracy


# img = x_test[3].reshape(200,200)
# decoded_img = model.predict(np.array([x_test[3].reshape(200,200)]))
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.imshow(img)
# plt.title('Test Image')
# plt.subplot(1, 3, 3)
# plt.imshow(decoded_img.squeeze())
# plt.title('Reconstructed Image')
# plt.show()
#
#
# img = x_train[3].reshape(200,200)
# decoded_img = model.predict(np.array([x_train[3].reshape(200,200)]))
# plt.figure(figsize=(15, 15))
# plt.subplot(1, 3, 1)
# plt.imshow(img)
# plt.title('Train Image')
# plt.subplot(1, 3, 3)
# plt.imshow(decoded_img.squeeze())
# plt.title('Reconstructed Image')
# plt.show()

#model.save_weights('autoencoder_weights_300.h5')
#encoder_model.save_weights('encoded_300.h5')
#decoder_model.save_weights('decoded.h5')

def plot_training_history(history):
    # Retrieve the loss and accuracy history from the training history dictionary
    loss_history = history['loss']
    val_loss_history = history['val_loss']
    accuracy_history = history['accuracy']
    val_accuracy_history = history['val_accuracy']

    # Plot the loss and accuracy history
    epochs = range(1, len(loss_history) + 1)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    ax1.plot(epochs, loss_history, label='Training Loss')
    ax1.plot(epochs, val_loss_history, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax2.plot(epochs, accuracy_history, label='Training Accuracy')
    ax2.plot(epochs, val_accuracy_history, label='Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.show()

#plot_training_history(history)
#
# plt.plot(history.history['accuracy'], label='train_loss')
# plt.plot(history.history['accuracy'], label='val_loss')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title("Training and Validation Accuracy")
# plt.legend()
# plt.show()

# def predictionModel()
# y_pred = np.argmax(y_pred_prob, axis=1) # choose class with highest probability
# accuracy = np.mean(y_pred == y_true) # calculate accuracy
# new_input = np.zeros((len(input_images),256))
# index =0
# for imgs in input_images:
#     encoded_img = encoder.predict(np.array([imgs.reshape(200, 200)]))
#     new_input[index]=encoded_img.reshape(256)
#     index+=1

# # Reshape the data into a 2D matrix with one row per image
# data = input_images.copy()
# n_images = data.shape[0]
# image_size = data.shape[1]
# data_2d = np.reshape(data, (n_images, image_size*image_size))
#
# # Create the MiniSom object
# som = MiniSom(n_neurons=10, m_neurons=10, input_len=image_size*image_size, sigma=1.5, learning_rate=0.5, neighborhood_function='gaussian')
#
# data = new_input.copy()
# n_images = data.shape[0]
# image_size = data.shape[1]
# # Initialization and training
# n_neurons = 9
# m_neurons = 9
# som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5,
#               neighborhood_function='gaussian', random_seed=0)
#
# som.pca_weights_init(data)
# som.train(data, 1000, verbose=True)
#
# data = input_images.copy()
# n_images = data.shape[0]
# image_size = data.shape[1]
# n_neurons = 9
# m_neurons = 9
# som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5,
#               neighborhood_function='gaussian', random_seed=0)
# data=data.reshape(len(data),40000)
# n_images = data.shape[0]
# image_size = data.shape[1]
# n_neurons = 20
# m_neurons = 20
# som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5,
#               neighborhood_function='', random_seed=0)
# # Train the SOM with the data
# som.train_random(data, 1000)
#
# # Plot the SOM
# som.plot_map()
