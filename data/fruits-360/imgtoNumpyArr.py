import os # to access directories, subdirectories, and files
import numpy as np
from PIL import Image # to access image data

# function to load and vectorize image data as numpy array
def convertImgToNumpyArr(image_path):
    try:
        img = Image.open(image_path)  # creating reference variable img to access image data
        img = img.resize((100, 100))  # resize image to 100x100 in case it is not
        img = np.array(img)  # converting image data from JPG to numpy array
        img = img / 255.0  # normalizing RGB values
        return img
    except Exception as e:
        print(f"Error processing image {image_path}: {e}") # error handling
        return None

# function to save each image data as numpy array
def saveImgAsNumpyArr(src_path, save_path):

    print(f"Original Image Folder: {src_path}")
    # creating the save folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Saving Image Folder: {save_path}")
    
    # going through each fruit folder in the source path (train or test)
    for fruit_folder in os.listdir(src_path):
        fruit_folder_path = os.path.join(src_path, fruit_folder)
        
        if os.path.isdir(fruit_folder_path): # checking if the path actually points to a folder

            print(f"Currently Processing Fruit Images of: {fruit_folder}")
            
            # going through each image file in the current folder
            for img in os.listdir(fruit_folder_path):
                img_path = os.path.join(fruit_folder_path, img)
                
                if img.endswith('.jpg'):  # checking if the file is an image
                    # print(f"Image file: {img_path}")
                    
                    img_as_numpy_arr = convertImgToNumpyArr(img_path)

                    if img_as_numpy_arr is None: # in case of error in image processing
                        print(f"Error processing image {img_path}")
                    else:
                        # creating similar img save path as the original img path in new save folder
                        subfolder_path = os.path.relpath(fruit_folder_path, src_path)

                        save_file_name = f"{os.path.splitext(img)[0]}.npy" # changing file extension to .npy
                        img_save_path = os.path.join(save_path, subfolder_path, save_file_name)
                        
                        # creating the save folder if it doesn't exist
                        os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                        np.save(img_save_path, img_as_numpy_arr)

# paths to the train and test folders
train_path = 'data/fruits-360/Training'
test_path = 'data/fruits-360/Test'

# Training Images to Numpy Arrays:
print("Processing Training Images")
saveImgAsNumpyArr(train_path, 'data/imgNumpyArr/Training')

# Testing Images to Numpy Arrays:
print("Processing Testing Images")
saveImgAsNumpyArr(test_path, 'data/imgNumpyArr/Testing')
