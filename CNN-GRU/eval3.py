import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GRU
from keras.layers import Activation, TimeDistributed, LSTM, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD  # Import SGD from keras.optimizers instead of tensorflow.keras.optimizers
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


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
batch_size = 24
size = (224,224,3)
img_width = img_hight = size[0]
clases = ['COVID19', 'NORMAL', 'PNEUMONIA']

#* Creating data generators
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = data_gen.flow_from_dataframe(Train_data, x_col='path', y_col='class',
                                          image_size=(img_hight, img_width), target_size=(
        img_hight, img_hight), color_mode='rgb',
                                          batch_size=batch_size, class_mode='categorical',
                                          classes=clases, subset='training')

val_data = data_gen.flow_from_dataframe(Train_data, x_col='path', y_col='class',
                                        image_size=(img_hight, img_width), target_size=(
        img_hight, img_hight), color_mode='rgb',
                                        batch_size=batch_size, class_mode='categorical',
                                        classes=clases, subset='validation')
test_data = data_gen.flow_from_dataframe(Test_data, x_col='path', y_col='class',
                                         image_size=(img_hight, img_width), target_size=(
        img_hight, img_hight), color_mode='rgb',
                                         batch_size=batch_size, class_mode='categorical',
                                         classes=clases, subset=None)

###################################UnSHuffled#########################################
# * will be used for the confusion matrix analysis for results

test_data_unshuffled = data_gen.flow_from_dataframe(Test_data, x_col='path', y_col='class',
                                                    image_size=(img_hight, img_width), target_size=(
        img_hight, img_hight), color_mode='rgb',
                                                    batch_size=batch_size, class_mode='categorical',
                                                    classes=clases, subset=None, shuffle=False)


#Model Architecture
def Create_model(Image_shape, block1=True, block2=True, block3=True,
                 block4=True, block5=True, lstm=True, regularizer=keras.regularizers.l2(0.0001),
                 Dropout_ratio=0.15):
    #Create the model
    model = keras.Sequential()

    #Configuring the inputshape
    model.add(keras.Input(shape=Image_shape))

    #First block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                     trainable=block1, kernel_regularizer=regularizer))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                     trainable=block1, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Second block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                     trainable=block2, kernel_regularizer=regularizer))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                     trainable=block2, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Third block
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                     trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                     trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Fourth block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block4, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block4, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block4, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Fifth block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block5, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block5, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block5, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add((MaxPooling2D(pool_size=(2, 2))))

    #Reshape the output of the last layer to be used in the GRU layer
    model.add(keras.layers.Reshape((7 * 7, 512)))
    model.add(GRU(512, activation='relu', trainable=lstm, return_sequences=True))
    model.add(BatchNormalization())

    #Flatten + Fc layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(Dropout_ratio))
    model.add(BatchNormalization())

    #Output layer
    #model.add(Dense(3, activation='linear'))
    model.add(Dense(3, activation='sigmoid'))
    return model

#Compile the model
def model_compiling(model, loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.0001)):  # Remove the decay argument
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )


model_path = "Model_2_Acc_based.h5"
checkpoint = keras.callbacks.ModelCheckpoint(  # Use keras.callbacks instead of tf.keras.callbacks
    filepath=model_path,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=15,
                          verbose=1,
                          restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=6,
                                            verbose=1,
                                            factor=0.2,
                                            min_lr=0.00000001)

model = Create_model(size)
model_compiling(model, loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.0001))

model.load_weights('model.h5')

# Load and preprocess the test image
image_path = 'C:/Users/dungu/Desktop/Model_Testing/PNEUMONIA/test8.jpg'
img = Image.open(image_path)
img = img.resize((img_width, img_hight))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add an additional dimension for batch size

# Get the model prediction for the test image
predictions = model.predict(img_array)
class_dict = {0: 'COVID19', 1: 'NORMAL', 2: 'PNEUMONIA'}
predicted_classes = [class_dict[i] for i in np.argmax(predictions, axis=1)]

# Display the image with the predicted classes and probabilities
plt.imshow(img)
plt.title(f"Predicted Class: {predicted_classes[0]}, Probability: {predictions[0][np.argmax(predictions[0])]:.2f}")
plt.axis('off')
plt.show()