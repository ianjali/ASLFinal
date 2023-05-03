import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
#import imgaug.augmenters as iaa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from EncoderDecoder import *
from sklearn.preprocessing import LabelEncoder
import pickle
#from tensorflow.keras.callbacks import EarlyStopping

train_dataset = 'Dataset/asl_alphabet_train/asl_alphabet_train/'
test_dataset = 'Dataset/asl_alphabet_test/asl_alphabet_test'

train_folder = os.listdir(train_dataset)
test_folder = os.listdir(test_dataset)
input_images = []
target_images = []
labels= []
test_img = []
test_labels = []
def preprocess(image_path,image_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = gray_img / 255.0
    return img_norm

for folder in train_folder:
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
        normalized= preprocess(image_path,image_size=(200,200))
        input_images.append(normalized)
        labels.append(target)
        cnt+=1
        if cnt == 500:
            break
        #target_images.append(target_norm)


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
    test_img.append(normalized)
    test_labels.append(target)

print(f"{'='*3} Reading dataset is finished {'='*3}")



#test_data = np.array(input_images).copy()
#test_data = test_data.reshape((len(test_data),256))

# test_data = np.array(input_images).copy()
# test_data = test_data.reshape((len(test_data),256))

# le = LabelEncoder()
# # Fit and transform the labels
# encoded_labels = le.fit_transform(labels)
# encoded_

#input_images = input_images
#target_images = target_images[:500]

# test_img = []
# test_labels = []
test_data = np.array(test_img)
#target_images = np.array(target_images)
input_images = np.array(input_images)
target_images = np.array(target_images)
target_labels = labels
target_test_labels =test_labels
le = LabelEncoder()
# Fit and transform the labels
target_labels = le.fit_transform(target_labels)
target_test_labels = le.fit_transform(target_test_labels)

x_train = input_images
x_test = test_data
y_train = tf.keras.utils.to_categorical(target_labels, num_classes=29)
y_test = tf.keras.utils.to_categorical(target_test_labels, num_classes=29)
#x_train, x_test, y_train,  y_test = train_test_split(input_images, input_images, test_size=0.2, random_state=42)
#x_train, x_test, y_train,  y_test = train_test_split(input_images, target_labels, test_size=0.2, random_state=42)
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=29)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=29)
history, model, encoder_model,decoder_model = trainEncoderDecoderWithTarget(x_train,y_train,x_test,y_test)


with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

model.save_weights('autoencoder_model_.h5')
encoder_model.save_weights('encoded_categori.h5')
decoder_model.save_weights('decoded_categori.h5')
# model,encoder = getEncoderDecoderWithTarget()
with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

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
accuracy_train = predictionEncoder(x_train,y_train,model)
accuracy_test = predictionEncoder(x_test, y_test,model)

#history1, model1, encoder_model1,decoder_model1 = trainEncoderDecoderWithTargetIncresedDimesnion(x_train,y_train,x_test,y_test)
x_train, x_val, y_train,  y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42,shuffle=True)
history1, model1, encoder_model1,decoder_model1 = trainEncoderDecoderWithTargetIncresedDimesnion(x_train,y_train,x_val,y_val)


# model.save_weights('autoencoder_model1_.h5')
# encoder_model.save_weights('encoded_categori1.h5')
# decoder_model.save_weights('decoded_categori1.h5')

x = np.arange(0, 40)
plt.plot(x,history1.history['loss'])
plt.plot(x,history1.history['val_loss'])
plt.title('Model  loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.xticks(range(min(x), max(x)+1))
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.tight_layout()
plt.show()