import cv2
import numpy as np
from google.colab import files

import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
from . import utils
# print('Base imported')

def test_func():
    print("Base Test function prints")

def printName():
    firstName = input("Please input your First Name: ")
    lastName = input("Please input your Last Name: ")
    print("\nNice to meet you, {} {}!".format(firstName, lastName))
    
def writeStory():
    word = ''
    story = ''
    while word.upper() != 'END':
        story = story + ' ' + word
        word = input("Please input next word in the story. Input END to finish the story: ")
    print('\n' + story)

def read_image(filepath):
  tmp_img = cv2.imread(filepath)
  if len(tmp_img.shape) == 3:
    tmp_img = tmp_img[:, :, ::-1]
  
  return tmp_img
    
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

def display_video(video):
    fig = plt.figure(figsize=(9,9))  #Display size specification

    mov = []
    for i in range(len(video)):  #Append videos one by one to mov
        img = plt.imshow(video[i], animated=True)
        plt.axis('off')
        mov.append([img])

    #Animation creation
    anime = animation.ArtistAnimation(fig, mov, interval=50, repeat_delay=1000)

    plt.close()
    return anime

