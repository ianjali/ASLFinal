import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import imgaug.augmenters as iaa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
def trainEncoderDecoder(x_train,y_train,x_test,y_test):
    encoder_units = [2048, 1024, 512]
    decoder_units = [ 512, 1024, 2048]
    input_shape = (200, 200)
    output_shape = (200, 200)
    # Define the encoder model
    encoder_input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(encoder_input)
    for units in encoder_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    encoder_output = tf.keras.layers.Dense(256, activation='relu')(x)
    encoder_model = tf.keras.Model(encoder_input, encoder_output)

    # Define the decoder model
    decoder_input = tf.keras.Input(shape=(256,))
    x = decoder_input
    for units in decoder_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    x = tf.keras.layers.Dense(tf.reduce_prod(output_shape), activation='relu')(x)
    decoder_output = tf.keras.layers.Reshape(output_shape)(x)
    decoder_model = tf.keras.Model(decoder_input, decoder_output)

    # Define the encoder-decoder model
    autoencoder_input = tf.keras.Input(shape=input_shape)
    encoded = encoder_model(autoencoder_input)
    decoded = decoder_model(encoded)
    autoencoder_model = tf.keras.Model(autoencoder_input, decoded)

    autoencoder_model.compile(optimizer='adam', loss='mse')
    # Train the model
    # Define early stopping criteria
    # earlystop = EarlyStopping(
    #     monitor='val_loss',
    #     patience=5,
    #     verbose=1,
    #     restore_best_weights=True
    # )
    autoencoder_model.load_weights('autoencoder_weights_50.h5')
    history = autoencoder_model.fit(x_train, y_train, epochs=50, batch_size=40, validation_data=(x_test, y_test))

    return history,autoencoder_model,encoder_model

def getEncoderDecoder():
    encoder_units = [2048, 1024, 512]
    decoder_units = [ 512, 1024, 2048]
    input_shape = (200, 200)
    output_shape = (200, 200)
    # Define the encoder model
    encoder_input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(encoder_input)
    for units in encoder_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    encoder_output = tf.keras.layers.Dense(256, activation='relu')(x)
    encoder_model = tf.keras.Model(encoder_input, encoder_output)

    # Define the decoder model
    decoder_input = tf.keras.Input(shape=(256,))
    x = decoder_input
    for units in decoder_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    x = tf.keras.layers.Dense(tf.reduce_prod(output_shape), activation='relu')(x)
    decoder_output = tf.keras.layers.Reshape(output_shape)(x)
    decoder_model = tf.keras.Model(decoder_input, decoder_output)

    # Define the encoder-decoder model
    autoencoder_input = tf.keras.Input(shape=input_shape)
    encoded = encoder_model(autoencoder_input)
    decoded = decoder_model(encoded)
    autoencoder_model = tf.keras.Model(autoencoder_input, decoded)

    autoencoder_model.compile(optimizer='adam', loss='mse')
    # Train the model
    # Define early stopping criteria
    # earlystop = EarlyStopping(
    #     monitor='val_loss',
    #     patience=5,
    #     verbose=1,
    #     restore_best_weights=True
    # )
    autoencoder_model.load_weights('autoencoder_weights_300.h5')
    encoder_model.load_weights('encoded_300.h5')

    plot_model(decoder_model, to_file='decoder_model_initial_layers.png', show_shapes=True, show_layer_names=True)
    #history = autoencoder_model.fit(x_train, y_train, epochs=50, batch_size=40, validation_data=(x_test, y_test))

    return autoencoder_model,encoder_model
def trainEncoderDecoderWithTarget(x_train,y_train,x_test,y_test):
    encoder_units = [2048, 1024, 512]
    decoder_units = [ 512, 1024, 2048]
    input_shape = (200, 200)
    #output_shape = (200, 200)
    num_classes = 29
    # Define the encoder model
    encoder_input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(encoder_input)
    for units in encoder_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    encoder_output = tf.keras.layers.Dense(256, activation='relu')(x)
    encoder_model = tf.keras.Model(encoder_input, encoder_output)

    # Define the decoder model
    decoder_input = tf.keras.Input(shape=(256,))
    x = decoder_input
    for units in decoder_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    decoder_output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    #x = tf.keras.layers.Dense(tf.reduce_prod(num_classes), activation='relu')(x)
    #decoder_output = tf.keras.layers.Reshape((num_classes,))(x)
    #decoder_label_output = tf.keras.layers.Dense(num_classes, activation='softmax')(decoder_output)
    decoder_model = tf.keras.Model(decoder_input, decoder_output)

    # Define the encoder-decoder model
    autoencoder_input = tf.keras.Input(shape=input_shape)
    encoded = encoder_model(autoencoder_input)
    decoded = decoder_model(encoded)
    autoencoder_model = tf.keras.Model(autoencoder_input, decoded)

    autoencoder_model.compile(optimizer='adam', loss='categorical_crossentropy')
    # Train the model
    # Define early stopping criteria
    # earlystop = EarlyStopping(
    #     monitor='val_loss',
    #     patience=5,
    #     verbose=1,
    #     restore_best_weights=True
    # )
    #autoencoder_model.load_weights('autoencoder_weights_categorical.h5')
    history = autoencoder_model.fit(x_train, y_train, epochs=100, batch_size=20, validation_data=(x_test, y_test))

    return history,autoencoder_model,encoder_model
def getEncoderDecoderWithTarget():
    encoder_units = [2048, 1024, 512]
    decoder_units = [ 512, 1024, 2048]
    input_shape = (200, 200)
    #output_shape = (200, 200)
    num_classes = 29
    # Define the encoder model
    encoder_input = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(encoder_input)
    for units in encoder_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    encoder_output = tf.keras.layers.Dense(256, activation='relu')(x)
    encoder_model = tf.keras.Model(encoder_input, encoder_output)

    # Define the decoder model
    decoder_input = tf.keras.Input(shape=(256,))
    x = decoder_input
    for units in decoder_units:
        x = tf.keras.layers.Dense(units, activation='relu')(x)
    decoder_output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    #x = tf.keras.layers.Dense(tf.reduce_prod(num_classes), activation='relu')(x)
    #decoder_output = tf.keras.layers.Reshape((num_classes,))(x)
    #decoder_label_output = tf.keras.layers.Dense(num_classes, activation='softmax')(decoder_output)
    decoder_model = tf.keras.Model(decoder_input, decoder_output)

    # Define the encoder-decoder model
    autoencoder_input = tf.keras.Input(shape=input_shape)
    encoded = encoder_model(autoencoder_input)
    decoded = decoder_model(encoded)
    autoencoder_model = tf.keras.Model(autoencoder_input, decoded)

    autoencoder_model.compile(optimizer='adam', loss='categorical_crossentropy')
    # Train the model
    # Define early stopping criteria
    # earlystop = EarlyStopping(
    #     monitor='val_loss',
    #     patience=5,
    #     verbose=1,
    #     restore_best_weights=True
    # )
    #autoencoder_model.load_weights('autoencoder_weights_categorical.h5')

    autoencoder_model.load_weights('autoencoder_category.h5')
    #encoder_model.load_weights('encoded_category.h5')
    plot_model(decoder_model, to_file='decoder_model_final_layers.png', show_shapes=True, show_layer_names=True)
    #history = autoencoder_model.fit(x_train, y_train, epochs=100, batch_size=20, validation_data=(x_test, y_test))

    return autoencoder_model,encoder_model
