# importing libraries and settings
from PIL import Image
from lang_sam import LangSAM
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

datagen = ImageDataGenerator(
    horizontal_flip = True,
)



# setting pu variables needed
correct_count = 0
wrong_count = 0
file_name = ""

y_true = []
y_score = []

model = LangSAM()



# extract names of pictures in a folder
filenames_list = []
directory = "/Users/eric/Desktop/aug"
files = os.listdir(directory)
files = [f for f in files if os.path.isfile(directory+'/'+f)] #Filtering only the files.
filenames_list.append(files)



# detecting 'marine debris' in each of the pictures given and getting the prediction value
for index, value in np.ndenumerate(filenames_list):
    file_name = value
    length_file_name = len(file_name)
    if file_name[length_file_name - 4:] == ".png":
        if file_name[0] == "o":
            y_true.append(0)
        else:
            y_true.append(1)
        image_pil = Image.open("/Users/eric/Desktop/aug/" + file_name).convert("RGB")
        text_prompt = "trash"
        masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
        print(file_name)
        print(logits)
        y_score.append(max(logits).item())
        print(max(logits).item)



# re-formatting list of prediction values to be compared with another list
y_score = np.array(y_score)
y_pred = y_score >= 0.5



# creating a confusion matrix and display on the screen
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()