import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, RepeatVector, Reshape
from keras.layers import Activation, TimeDistributed, LSTM, BatchNormalization, ConvLSTM2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD,Adam  
import tensorflow as tf
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

model.load_weights(model_path)

# Load the model history
History = pd.read_csv('lstm2.csv')
History.insert(0, 'epochs', range(1, len(History) + 1))
History.head()

# Plotting the graphs for the model history
def plot_model_history(history):
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['epochs'], history['loss'], label='Train Loss')
    plt.plot(history['epochs'], history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Train and Validation Loss')

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['epochs'], history['accuracy'], label='Train Accuracy')
    plt.plot(history['epochs'], history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train and Validation Accuracy')

    plt.tight_layout()
    plt.show()

    # Learning rate plot
    plt.figure(figsize=(8, 4))
    plt.plot(history['epochs'], history['lr'], label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('Learning Rate Schedule')
    plt.tight_layout()
    plt.show()

# Call the function to plot the model history
plot_model_history(History)