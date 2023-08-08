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
from keras.optimizers import SGD
import seaborn as sns
import matplotlib.pyplot as plt


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


def Create_model(Image_shape, block1=True, block2=True, block3=True,
                 block4=True, block5=True, lstm=True, regularizer=keras.regularizers.l2(0.0001),
                 Dropout_ratio=0.15):
    # * Create the model
    model = keras.Sequential()

    # * configure the inputshape
    model.add(keras.Input(shape=Image_shape))

    # * Add the first block
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                     trainable=block1, kernel_regularizer=regularizer))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                     trainable=block1, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the second block
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                     trainable=block2, kernel_regularizer=regularizer))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                     trainable=block2, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the third block
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                     trainable=block3, kernel_regularizer=regularizer))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                     trainable=block3, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the fourth block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block4, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block4, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block4, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # * Add the fifth block
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block5, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block5, kernel_regularizer=regularizer))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
                     trainable=block5, kernel_regularizer=regularizer))
    model.add(BatchNormalization())
    model.add((MaxPooling2D(pool_size=(2, 2))))

    # * Reshape the output of the last layer to be used in the GRU layer
    model.add(keras.layers.Reshape((7 * 7, 512)))
    model.add(GRU(512, activation='relu', trainable=lstm, return_sequences=True))
    model.add(BatchNormalization())

    # * flatten + Fc layer
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(Dropout_ratio))
    model.add(BatchNormalization())

    # * Output layer
    # model.add(Dense(3, activation='linear'))
    model.add(Dense(3, activation='sigmoid'))
    return model


# * compile function
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
predicted_classes = [class_dict.get(list(predictions[i]).index(predictions[i].max())) for i in range(len(predictions))]
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