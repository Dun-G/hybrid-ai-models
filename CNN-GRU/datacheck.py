import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

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


# Function to display one image per class
def display_one_image_per_class(dataframe):
    unique_classes = dataframe['class'].unique()
    for class_name in unique_classes:
        class_df = dataframe[dataframe['class'] == class_name]
        img_path = class_df.iloc[0]['path']
        img_class = class_df.iloc[0]['class']
        img = Image.open(img_path)
        plt.imshow(img, cmap='gray')
        plt.title(f"Class: {img_class}")
        plt.axis('off')  # To remove axes and ticks
        plt.show()


# Function to check class distribution and plot a bar chart
def check_class_distribution(dataframe, data_name):
    class_distribution = dataframe['class'].value_counts()
    print(f"Class Distribution for {data_name}:")
    print(class_distribution)
    plt.bar(class_distribution.index, class_distribution.values)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'Class Distribution for {data_name}')
    plt.show()


# Function to check image properties (resolution)
def check_image_properties(dataframe):
    for idx, row in dataframe.iterrows():
        img_path = row['path']
        img = Image.open(img_path)
        width, height = img.size
        print(f"Image {idx + 1}:")
        print(f"  Resolution: {width} x {height}")
        print()


# Function to check for corrupted images and prompt the user
def check_corrupted_images(dataframe):
    corrupted_images = []
    for idx, row in dataframe.iterrows():
        img_path = row['path']
        try:
            img = Image.open(img_path)
            img.verify()  # Raises an exception if the image is corrupted
        except (IOError, SyntaxError) as e:
            corrupted_images.append(img_path)

    if corrupted_images:
        print("Corrupted images found:")
        for img_path in corrupted_images:
            print(img_path)
        print("Please remove or fix these images.")
    else:
        print("No corrupted images found in the dataset.")


# Function to check the DataFrame shape
def check_dataframe_shape(dataframe, data_name):
    print(f"{data_name} DataFrame Shape:")
    print(dataframe.shape)

# Function to display sample rows of the DataFrame
def display_sample_rows(dataframe, data_name, num_rows=5):
    print(f"Sample rows of {data_name} DataFrame:")
    print(dataframe.head(num_rows))


# Display one image per class from Train_data
display_one_image_per_class(Train_data)

# Display one image per class from Test_data
display_one_image_per_class(Test_data)

# Check class distribution in Train_data and Test_data
check_class_distribution(Train_data, 'Train data')
check_class_distribution(Test_data, 'Test data')

# Check image properties (resolution) in Train_data and Test_data
check_image_properties(Train_data)
check_image_properties(Test_data)

# Check for corrupted images in Train_data and Test_data
check_corrupted_images(Train_data)
check_corrupted_images(Test_data)

# Check the DataFrame shape for Train_data and Test_data
check_dataframe_shape(Train_data, 'Train')
check_dataframe_shape(Test_data, 'Test')

# Display sample rows of the DataFrames
display_sample_rows(Train_data, 'Train')
display_sample_rows(Test_data, 'Test')

# Additional information about the dataset
print("Number of images in the train dataset:", len(Train_data))
print("Number of images in the test dataset:", len(Test_data))
print("Number of unique classes in the train dataset:", len(Train_data['class'].unique()))
print("Number of unique classes in the test dataset:", len(Test_data['class'].unique()))