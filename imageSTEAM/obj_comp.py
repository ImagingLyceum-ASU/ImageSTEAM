import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from google.colab import files

import os, random
import zipfile
from shutil import copyfile


# Loads the images into the computer memory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

# Displays an image, converts BGR to RGB
def display_img(a, title = "Original"):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    plt.imshow(a), plt.title(title)
    plt.show()

# Displays an RGB image
def display_rgb_img(a, title = "Original"):
    plt.imshow(a), plt.title(title)
    plt.show()

# Resizes to the smallest image height
def resize_simple(im1, im2):
	if len(im1) > len(im2):
		im1 = cv2.resize(im1, (len(im2[0]), len(im2)))
	else:
		im2 = cv2.resize(im2, (len(im1[0]), len(im1)))
	return (im1, im2)

# Displays all of the images in a folder
def display_folder(dir):
  for filename in os.listdir(dir):
    temp_img = mpimg.imread(dir + '/' + filename)
    display_rgb_img(temp_img, filename)

def pick_random_image(dir):
  im_name = random.choice(os.listdir(dir))
  print(im_name)
  return cv2.imread(dir + im_name)

def extract_images():
  local_zip = '/tmp/cat_pic.zip' #get zip file
  zip_ref = zipfile.ZipFile(local_zip, 'r') #create zifile object to read the zipfile
  zip_ref.extractall('/content') #extract content into /content folder

  local_zip = '/tmp/dog_pic.zip'
  zip_ref = zipfile.ZipFile(local_zip, 'r')
  zip_ref.extractall('/content')

  zip_ref.close() #close zip

  copyfile('/content/dog_pic/dog1.jpg', '/content/dog1.jpg') #copy
  copyfile('/content/cat_pic/cat1.jpg', '/content/cat1.jpg')
  copyfile('/content/dog_pic/dog2.jpg', '/content/dog2.jpg') #copy
  copyfile('/content/cat_pic/cat2.jpg', '/content/cat2.jpg')
  copyfile('/content/dog_pic/dog3.jpg', '/content/dog3.jpg') #copy
  copyfile('/content/cat_pic/cat3.jpg', '/content/cat3.jpg')
  copyfile('/content/dog_pic/dog4.jpg', '/content/dog4.jpg') #copy
  copyfile('/content/cat_pic/cat4.jpg', '/content/cat4.jpg')
  copyfile('/content/dog_pic/dog5.jpg', '/content/dog5.jpg') #copy
  copyfile('/content/cat_pic/cat5.jpg', '/content/cat5.jpg')
  
  print('Example images to choose from:')
  imgs = (os.listdir('/content'))
  for img in imgs:
    if '.jpg' in img:
      print(img)


def approximate_gradient(dx, dy):
  gradient = np.hypot(dx, dy)
  gradient = gradient / np.max(gradient)
  gradient = (gradient * 255).astype(np.uint8)
  plt.imshow(gradient, cmap='gray')


def cos_similarity(a, b):
  return np.dot(a, b)

def is_it_a_dog(dog_gradient, cat_gradient):
  g_dog = dog_gradient
  g_cat = cat_gradient
  #get new dog image to compare
  compare_dog = pick_random_image('/content/dog_pic/')
  compare_dog = cv2.cvtColor(compare_dog, cv2.COLOR_BGR2GRAY)
  plt.imshow(compare_dog, cmap='gray'), plt.title('Grayscale')
  plt.show()

  #get approximate gradient of new dog image
  dx_compare_dog = cv2.filter2D(compare_dog, -1, sobel_x)
  dy_compare_dog = cv2.filter2D(compare_dog, -1, sobel_y)
  g_compare_dog = approximate_gradient(dx_compare_dog, dy_compare_dog)

  #Resize the images
  g_dog, g_compare_dog = resize_simple(g_dog, g_compare_dog)
  #Dot Product to compare new dog image to previous dog image
  dog_dog_dot = cos_similarity(g_dog.flatten(), g_compare_dog.flatten())
  print("Random dog and dog1 dot product: ", dog_dog_dot)

  #compare new dog image to cat image
  g_cat, g_compare_dog = resize_simple(g_cat, g_compare_dog)
  cat_dog_dot = cos_similarity(g_cat.flatten(), g_compare_dog.flatten())
  print("Random dog and cat1 dot product: ", cat_dog_dot)

  if (dog_dog_dot > cat_dog_dot): 
    print("It might be a dog")
  elif (dog_dog_dot == cat_dog_dot):
    print("It might be both a cat and a dog?")
  else:
    print("It might be a cat")

def is_it_a_cat(cat_gradient, dog_gradient):
  g_cat = cat_gradient
  g_dog = dog_gradient

  #Get new random cat image
  compare_cat = pick_random_image('/content/cat_pic/') #encapsulate a lot of code to be easier to consume
  compare_cat = cv2.cvtColor(compare_cat, cv2.COLOR_BGR2GRAY)
  plt.imshow(compare_cat, cmap='gray'), plt.title('Grayscale')
  plt.show()

  # Canny Image of the New Image
  dx_compare_cat = cv2.filter2D(compare_cat, -1, sobel_x)
  dy_compare_cat = cv2.filter2D(compare_cat, -1, sobel_y)
  g_compare_cat = approximate_gradient(dx_compare_cat, dy_compare_cat)

  #Resize the images
  g_cat, g_compare_cat = resize_simple(g_cat, g_compare_cat)
  #Dot Product to compare new cat to previous cat
  cat_cat_dot = cos_similarity(g_cat.flatten(), g_compare_cat.flatten())
  print("Random cat and cat1 dot product: ", cat_cat_dot)

  #Compare new cat to dog
  g_dog, g_compare_cat = resize_simple(g_dog, g_compare_cat)
  dog_cat_dot = cos_similarity(g_dog.flatten(), g_compare_cat.flatten())
  print("Random cat and dog1 dot product: ", dog_cat_dot)

  if (cat_cat_dot > dog_cat_dot):
    print("It might be a cat")
  elif (cat_cat_dot == dog_cat_dot):
    print("It might be both a cat and a dog?")
  else:
    print("It might be a dog")
