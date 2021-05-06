import cv2
import numpy as np
from google.colab import files

from . import utils
# print('Base imported')

def test_func():
    print("Base Test function prints")

def printName():
    firstName = input("Please input your First Name")
    lastName = input("Please input your Last Name")
    print("Nice to meet you, {} {}!".format(firstName, lastName))
    
def writeStory():
    word = ''
    story = ''
    while word.upper() != 'END':
        story = story + ' ' + word
        word = input("Please input next word in the story. Input END to finish the story")
    print(story)
    
def upload_image():
    uploaded = files.upload()

    files_list = []
    images = []
    for fn in uploaded.keys():
        print('User uploaded file "{name}" with length {length} bytes'.format(
            name=fn, length=len(uploaded[fn])))
        files_list.append(fn)

    for filename in files_list:
        temp_img = cv2.imread(filename)
        if len(temp_img.shape) == 3:
            temp_img = temp_img[:, :, ::-1]
            
        utils.display_img(temp_img)
        
        if len(files_list) > 1:
            images.append(temp_img)
        else:
            images = temp_img
            
    return images