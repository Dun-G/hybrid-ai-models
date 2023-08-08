import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,GRU, Reshape, ConvLSTM2D
from keras.layers import Activation, TimeDistributed, LSTM, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD,Adam  # Import SGD from keras.optimizers instead of tensorflow.keras.optimizers
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
import seaborn as sns


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

# * compile function
def model_compiling(model, loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001)):
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    
# Define the model
model = Create_model(input_shape=(img_hight, img_width, 3), lstm_units=128)  # Corrected parameter

# Compile the model
model_compiling(model, loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001))

# Display the model summary
print(model.summary())

# Save the model to a file
model_path = "lstm_new.h5"
model.save(model_path)

# Load the saved model with the correct file name
model = keras.models.load_model(model_path)

# Evaluate the model on validation data
val_evaluation = model.evaluate(val_data)
print("Validation Data Evaluation:")
print("Loss:", val_evaluation[0])
print("Accuracy:", val_evaluation[1])
print()

# Evaluate the model on the test data
test_evaluation = model.evaluate(test_data)
print("Test Data Evaluation:")
print("Loss:", test_evaluation[0])
print("Accuracy:", test_evaluation[1])
print()

# Evaluate the model on the unshuffled test data
test_unshuffled_evaluation = model.evaluate(test_data_unshuffled)
print("Unshuffled Test Data Evaluation:")
print("Loss:", test_unshuffled_evaluation[0])
print("Accuracy:", test_unshuffled_evaluation[1])
print()

# predictions of the model on the unshuffled test set
predictions = model.predict(test_data_unshuffled, verbose=1, use_multiprocessing=False)
predictions.shape

class_dict = test_data.class_indices
class_dict = {value: key for key, value in class_dict.items()}
predicted_classes = [class_dict.get(list(predictions[i]).index(max(predictions[i]))) for i in range(len(predictions))]
Test_data['predicted_class'] = predicted_classes
Test_data['matched'] = (Test_data['class'] == Test_data['predicted_class'])

# Create a new DataFrame to store the prediction information
prediction_info = pd.DataFrame({
    'Actual Class': Test_data['class'],
    'Predicted Class': Test_data['predicted_class'],
    'Matched': Test_data['matched']
})

# Display the prediction information table
print(prediction_info)
print("Correct prediction: ", len(Test_data[Test_data['matched'] == True]))
print("False prediction: ", len(Test_data[Test_data['matched'] == False]))
print("Accuracy of the model on unseen data: ", round(len(Test_data[Test_data['matched'] == True]) / len(Test_data), 4), "%")

# Create a confusion matrix
confusion_matrix = pd.crosstab(Test_data['class'], Test_data['predicted_class'], rownames=['Actual'], colnames=['Predicted'])

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Calculate precision, recall, and F1 score
true_positives = len(Test_data[Test_data['matched'] == True])
false_positives = len(Test_data[(Test_data['matched'] == False) & (Test_data['predicted_class'] != 'NORMAL')])
false_negatives = len(Test_data[(Test_data['matched'] == False) & (Test_data['predicted_class'] == 'NORMAL')])

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)

# Print accuracy, precision, recall, and F1 score
print("Accuracy of the model on unseen data: {:.2f}%".format(test_evaluation[1] * 100))
print("Precision of the model on unseen data: {:.2f}%".format(precision * 100))
print("Recall of the model on unseen data: {:.2f}%".format(recall * 100))
print("F1 score of the model on unseen data: {:.2f}".format(f1_score))