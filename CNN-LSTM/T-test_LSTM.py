import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, RepeatVector, Reshape
from keras.layers import Activation, TimeDistributed, LSTM, BatchNormalization, ConvLSTM2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD,Adam  
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import os


# Reading main directory and creating dataframes for train and test data
directory = 'C:/Users/dungu/Documents/Pycharm Dataset/Data'
Train_data = pd.DataFrame(columns=['path', 'class'])
Test_data = pd.DataFrame(columns=['path', 'class'])

for filename in os.listdir(directory):
    for filename2 in os.listdir(directory + '/' + filename):
        for images in os.listdir(directory + '/' + filename + '/' + filename2):
            if filename == 'train':
                Train_data = pd.concat([Train_data, pd.DataFrame(
                    {'path': [directory + '/' + filename + '/' + filename2 + '/' + images], 'class': [filename2]})],
                                       ignore_index=True)
            else:
                Test_data = pd.concat([Test_data, pd.DataFrame(
                    {'path': [directory + '/' + filename + '/' + filename2 + '/' + images], 'class': [filename2]})],
                                      ignore_index=True)

#* Defining parameters
learning_rate = 5e-6
batch_size = 24
size = (224,224,3)
img_width = img_hight = size[0]
clases = ['COVID19', 'NORMAL', 'PNEUMONIA']

# Separate data for validation and test
train_data, val_data = train_test_split(Train_data, test_size=0.2, random_state=42)

# Creating data generators for training, validation, and test data
data_gen = ImageDataGenerator(rescale=1./255)
batch_size = 24

train_data_gen = data_gen.flow_from_dataframe(train_data, x_col='path', y_col='class',
                                              target_size=(img_hight, img_width), color_mode='rgb',
                                              batch_size=batch_size, class_mode='categorical',
                                              classes=clases, shuffle=True)

val_data_gen = data_gen.flow_from_dataframe(val_data, x_col='path', y_col='class',
                                            target_size=(img_hight, img_width), color_mode='rgb',
                                            batch_size=batch_size, class_mode='categorical',
                                            classes=clases, shuffle=False)

test_data_gen = data_gen.flow_from_dataframe(Test_data, x_col='path', y_col='class',
                                             target_size=(img_hight, img_width), color_mode='rgb',
                                             batch_size=batch_size, class_mode='categorical',
                                             classes=clases, shuffle=False)

###################################UnSHuffled#########################################
# * will be used for the confusion matrix analysis for results

test_data_unshuffled_gen = data_gen.flow_from_dataframe(Test_data, x_col='path', y_col='class',
                                                        target_size=(img_hight, img_width), color_mode='rgb',
                                                        batch_size=batch_size, class_mode='categorical',
                                                        classes=clases, shuffle=False)

def Create_model(input_shape, lstm_units=128, dropout_rate=0.25):
    # Create the model
    model = keras.Sequential()

    # CNN Part - Block 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))

    # CNN Part - Block 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))

    # CNN Part - Block 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))

    # CNN Part - Block 4
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))

    # CNN Part - Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rate))

    # Reshape for LSTM
    model.add(Reshape((-1, 7, 7, 512)))  # Reshape for ConvLSTM2D input

    # LSTM Part
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', return_sequences=True))
    model.add(Dropout(dropout_rate))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(3, activation='softmax'))

    return model

# Compile function
def model_compiling(model, loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001)):
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

model = Create_model(size)
model_compiling(model, loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.0001))

model_path = "lstm_new.h5"

# Load the trained model from the path
model = keras.models.load_model(model_path)

def preprocess_input(x):
    x = x / 255.0  # Rescale pixel values to [0, 1]
    return x

def evaluate_model(model, data_generator, num_images_per_class=20):
    results = []

    for class_name in clases:
        class_data = Test_data[Test_data['class'] == class_name].sample(num_images_per_class)
        for _, row in class_data.iterrows():
            image_path = row['path']
            true_class = row['class']

            # Load and preprocess the image
            img = image.load_img(image_path, target_size=(img_hight, img_width))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x)[0]
            predicted_class = clases[np.argmax(prediction)]

            true_positive = int(predicted_class == true_class and predicted_class == class_name)
            false_positive = int(predicted_class != true_class and predicted_class == class_name)
            true_negative = int(predicted_class != true_class and predicted_class != class_name)
            false_negative = int(predicted_class == true_class and predicted_class != class_name)
            accuracy = true_positive / num_images_per_class

            results.append([true_positive, true_negative, false_positive, false_negative, accuracy])

    columns = ['TP', 'TN', 'FP', 'FN', 'Accuracy']
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv("LSTM_Results.csv", index_label="Image Number")

# Load the trained model from the path
model = keras.models.load_model(model_path)

# Evaluate the model on test data and save results to CSV file
evaluate_model(model, test_data_gen, num_images_per_class=20)