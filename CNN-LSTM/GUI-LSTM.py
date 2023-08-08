import pandas as pd
import numpy as np
import pygame
import os
import warnings
import threading
import time
import pyttsx3
warnings.filterwarnings('ignore')
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, filedialog
from ttkbootstrap import Style
from PIL import Image, ImageTk
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ConvLSTM2D, Reshape
from keras.layers import Activation, TimeDistributed, LSTM, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam  # Import SGD from keras.optimizers instead of tensorflow.keras.optimizers


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
    
# Save the model to a file
model_path = "lstm_new.h5"

# Load the saved model with the correct file name
model = keras.models.load_model(model_path)

def load_model():
    model = Create_model(size)
    model_compiling(model, loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001))
    model_path = os.path.join(os.getcwd(), 'lstm_new.h5')
    model.load_weights(model_path)
    return model

class PneumoniaCovidDetectorApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Pneumonia and COVID-19 Detector")
        self.root.geometry("500x500")

        self.style = ttk.Style()
        self.style.configure('Transparent.TEntry', background='', border='0')  # Adjust style for text-boxes
        self.model = model

        # Create a canvas to display the background image
        self.canvas = tk.Canvas(self.root, width=500, height=500)
        self.canvas.pack()

        # Set the canvas to expand and fill the entire window
        self.canvas.pack_propagate(False)

        # Load the background image and resize it to fit the window size
        self.bg_image = Image.open("back.png")
        self.bg_image = self.bg_image.resize((500, 500), Image.LANCZOS)  # Use LANCZOS resampling
        self.bg_image_tk = ImageTk.PhotoImage(self.bg_image)

        # Display the background image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_image_tk)

        self.create_widgets()  # Create the widgets on top of the canvas

        self.engine = pyttsx3.init('sapi5')

        # Load voice files
        self.class_to_voice = {
            'COVID19': 'covid.mp3',
            'NORMAL': 'normal.mp3',
            'PNEUMONIA': 'pnm.mp3'
        }

    def set_styles(self):
        # Set custom styles for widgets
        self.style.configure("TLabel", foreground="black",
                             background=self.root.cget('bg'))  # Set background as window bg
        self.style.configure("TButton", font=("Helvetica", 12), background="lightblue")
        self.style.configure("Horizontal.TProgressbar", thickness=10, troughcolor="gray", background="blue")
        self.style.configure("Prediction.TLabel", font=("Helvetica", 16), foreground="green",
                             background=self.root.cget('bg'))
        self.style.configure("Warning.TLabel", font=("Helvetica", 16), foreground="red",
                             background=self.root.cget('bg'))
        self.style.configure("Primary.TLabel", font=("Helvetica", 14), foreground="black",
                             background=self.root.cget('bg'))

        # Remove the text background and set the text color to white for the "Upload a Chest X-Ray Image" label
        self.style.configure("Upload.TLabel", foreground="white", background=self.root.cget('bg'))

    def set_background_image(self):
        # Load the background image and resize it to fit the window size
        bg_image = Image.open("back.png")
        bg_image = bg_image.resize((500, 500), Image.LANCZOS)  # Use LANCZOS resampling

        # Convert the image to PhotoImage format
        self.bg_image_tk = ImageTk.PhotoImage(bg_image)

        # Display the background image on the canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_image_tk)

    def create_widgets(self):
        # Create the "Upload a Chest X-Ray Image" label with white text and transparent background
        self.label = tk.Label(self.canvas, text="Upload a Chest X-Ray Image", font=("Helvetica", 16), fg="white", bg=self.root.cget('bg'))
        self.label.pack(pady=10)

        self.load_button = ttk.Button(self.canvas, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.process_button = ttk.Button(self.canvas, text="Process Image", command=self.process_image)
        self.process_button.pack(pady=5)

        self.output_label = ttk.Label(self.canvas, text="", style="primary.TLabel", font=("Helvetica", 14), justify='center')
        self.output_label.pack(pady=5)

        # Use determinate mode for the progress bar
        self.progress_bar = ttk.Progressbar(self.canvas, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        if file_path:
            self.img = Image.open(file_path).resize((224, 224))  # Resize the image to (224, 224)
            self.img_tk = ImageTk.PhotoImage(self.img)
            self.label.config(image=self.img_tk)

    def process_image(self):
        if hasattr(self, 'img'):
            self.output_label.config(text="Processing...", style="warning.TLabel")
            self.progress_bar["value"] = 0  # Reset the progress bar
            self.root.update()

            img_array = np.array(self.img)  # Convert image to array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize pixel values

            # Perform image processing in a separate thread
            thread = threading.Thread(target=self.process_image_in_thread, args=(img_array,))
            thread.start()

    def process_image_in_thread(self, img_array):
        prediction = self.model.predict(img_array)
        predicted_class = clases[np.argmax(prediction)]
        probability = prediction[0][np.argmax(prediction)] * 100

        # Get the predicted class and probability
        predicted_class = clases[np.argmax(prediction)]
        probability = prediction[0][np.argmax(prediction)] * 100

        # Update the progress bar while processing
        for i in range(100):
            self.progress_bar["value"] = i
            self.root.update()
            time.sleep(0.02)

        # Switch to indeterminate mode for animation after reaching 100%
        self.progress_bar["mode"] = "indeterminate"
        self.progress_bar["value"] = 100

        # Call the function to speak the output after a delay
        self.root.after(1000, self.play_voice_file, predicted_class)

        # Wait for a moment before speaking the probability
        time.sleep(1)

        # Call the function to speak the probability after a delay
        probability_str = f"{probability:.2f}%"
        self.root.after(3000, self.speak_output, probability_str)

        # Set the progress bar back to 0 and determinate mode
        self.progress_bar["value"] = 0
        self.progress_bar["mode"] = "determinate"

        self.output_label.config(text=f"Predicted Class: {predicted_class}\nProbability: {probability:.2f}%",
                                 style="success.TLabel")

    def play_voice_file(self, predicted_class):
        # Check if the predicted class has a corresponding voice file
        if predicted_class in self.class_to_voice:
            voice_file = self.class_to_voice[predicted_class]
            pygame.mixer.init()
            pygame.mixer.music.load(voice_file)
            pygame.mixer.music.play()

    def speak_output(self, text):
        # Check if the predicted class has a corresponding voice file
        if text in self.class_to_voice:
            voice_file = self.class_to_voice[text]
            pygame.mixer.init()
            pygame.mixer.music.load(voice_file)
            pygame.mixer.music.play()

        # Use pyttsx3 to speak the output
        self.engine.say(text)
        self.engine.runAndWait()

if __name__ == "__main__":
    model = load_model()

    root = tk.Tk()
    app = PneumoniaCovidDetectorApp(root, model)
    app.set_background_image()

    root.mainloop()